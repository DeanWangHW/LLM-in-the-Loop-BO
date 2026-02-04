"""注释：超参数调优（HPT）任务的 LLM 代理与候选采样/建模逻辑。"""

import openai  # OpenAI 官方 SDK
import random  # 随机打乱与采样
import json  # JSON 解析与生成
from tqdm import trange  # 进度条工具
from scipy.stats import norm  # 正态分布函数，用于 EI 计算
import numpy as np  # 数值计算
import torch  # 张量计算与 GP 训练
from collections import defaultdict  # 便于按 key 聚合
from multiprocessing import Pool, cpu_count  # 多进程并行
from tqdm import trange  # 进度条工具（重复导入，保留原逻辑）
from botorch.acquisition import UpperConfidenceBound  # UCB 采集函数
from helper_func import *  # 引入项目内工具函数

# functions used to make LLAMBO parallelly
def _sample_one_candidate_HPT(args):
    """注释：根据历史超参表现生成新候选配置。"""
    i, history_variant_str, func_desc, target_score = args  # 解包参数
    prompt = f"""  # 构造提示词
    The following are examples of the performance of a {func_desc['md_name']} measured in mean square error and the corresponding model hyperparameter configurations. 
    {history_variant_str}
    The model is evaluated on a regression task. {func_desc['data_desc']}
    The allowable ranges for the hyperparameters are: {func_desc['md_param']}. 
    Recommend a configuration that can achieve the target mean square error of {target_score}, and each dimension must strictly within the allowable range specified above.  
    Return only a single {func_desc['md_ndim']}-dimensional numerical vector with the highest possible precision. 
    The response need to be a list and must be strictly valid JSON. 
    Do not include any explanations, labels, formatting, or extra text like jsom. 
    """
    from openai import OpenAI  # ensure import in subprocess
    import json  # 子进程内确保 JSON 可用

    client = OpenAI()  # 初始化 OpenAI 客户端

    while True:  # 直到解析成功
        try:  # 捕获 JSON/类型错误
            messages = [  # 构造对话消息
                {
                    "role": "system",
                    "content": "You are an AI assistant that helps me maximizing the accuracy by tuning the hyperparameter in the machine learning model.",
                },
                {"role": "user", "content": prompt},
            ]
            response = client.chat.completions.create(  # 发起 LLM 调用
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=50,
            ).choices[0].message.content.strip()  # 取回复文本
            extracted_value = json.loads(response)  # 解析 JSON
            extracted_value = [np.float64(v) for v in extracted_value]  # 转 float64
            return tuple(extracted_value)  # 返回候选配置

        except (ValueError, json.JSONDecodeError):  # 解析失败则重试
            print("Invalid LLM sampling response, retrying...")  # 提示
            continue  # 继续循环
                
def _predict_llm_score_HPT(args):
    """注释：预测给定超参配置的性能（MSE）。"""
    x, history_variant_str, func_desc = args  # 解包参数
    if func_desc["md_name"] == "Random Forest":  # 随机森林超参格式
        pred_card = f""""(max_depth, min_samples_split, min_samples_leaf, max_features): {x}"""
    elif func_desc["md_name"] == "Support Vector Regression":  # SVR 超参格式
        pred_card = f""""(C, epsilon, gamma): {x}"""
    elif func_desc["md_name"] == "XGBoost":  # XGBoost 超参格式
        pred_card = f""""(max_depth, learning_rate, subsample, colsample_bytree): {x}"""
    elif func_desc["md_name"] == "Neural Net":  # 神经网络超参格式
        pred_card = f""""(hidden_layer_sizes, alpha, learning_rate_init): {x}"""
        
    prompt = f"""
    The following are examples of the performance of a {func_desc['md_name']} measured in mean square error and the corresponding model hyperparameter configurations. 
    {history_variant_str}     
    The model is evaluated on a regression task. {func_desc['data_desc']}
    {func_desc['data_desc']}
    The dataset contains {func_desc['data_nsamp']} samples and {func_desc['data_nfeature']} features and all of the features are continuous. 
    Predict the mean square error when the model hyperparameter configurations is set to be {pred_card}. Do not include any explanations, labels, formatting, or extra text. The response must be strictly a valid floating-point number.
    """

    import json  # 子进程内确保 JSON 可用
    from openai import OpenAI  # 子进程内确保 OpenAI 可用
    client = OpenAI()  # 初始化 OpenAI 客户端
    while True:  # 直到成功解析数值
        try:  # 捕获数值转换错误
            messages = [  # 构造对话消息
                {
                    "role": "system",
                    "content": "You are an AI assistant that helps me maximizing the accuracy by tuning the hyperparameter in the machine learning model.",
                },
                {"role": "user", "content": prompt},
            ]
            response = client.chat.completions.create(  # 调用 LLM
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=50,
            ).choices[0].message.content.strip()  # 取回复文本
            return float(response), tuple(x)  # 返回预测值与配置

        except ValueError:  # 解析失败则重试
            print("Invalid LLM selecting response, retrying...")  # 提示
            continue  # 继续循环

def build_gp_model(args):
    """注释：基于扩展历史构建单个 GP 模型（供并行）。"""
    import torch  # 子进程内导入 torch
    from gpytorch.mlls import ExactMarginalLogLikelihood  # 似然
    from botorch.models import SingleTaskGP  # GP 模型
    from botorch.fit import fit_gpytorch_mll  # 拟合工具

    next_x_LLM, sample_val, history, lower_bounds, upper_bounds = args  # 解包参数

    extended_history = history + [(tuple(next_x_LLM), sample_val)]  # 扩展历史
    X = torch.tensor([list(x) for x, _ in extended_history], dtype=torch.double)  # 输入张量
    X_scaled = (X - lower_bounds) / (upper_bounds - lower_bounds)  # 归一化到 [0,1]
    Y = torch.tensor([[y] for _, y in extended_history], dtype=torch.double)  # 输出张量

    model = SingleTaskGP(X_scaled, Y)  # 构建 GP
    mll = ExactMarginalLogLikelihood(model.likelihood, model)  # 似然对象
    fit_gpytorch_mll(mll)  # 拟合模型

    return model  # 返回模型
# LLAMBO agent function
class LLAMAGENT_HPT:
    """注释：HPT 版本 LLAMBO 代理。"""

    def __init__(self, history, func_desc, alpha=0.1, num_cand=10, max_surrogate_eval=10):
        self.alpha = alpha  # 目标分数插值系数
        self.history = [(tuple(x), y) for x, y in history]  # 历史记录（保证可哈希）
        self.grid_results = {}  # 缓存代理模型结果
        self.num_cand = num_cand  # 候选数量
        self.func_desc = func_desc  # 任务描述字典
        self.max_surrogate_eval = max_surrogate_eval  # 代理评估次数
    
    def query_llm(self, prompt, model="gpt-3.5-turbo", max_tokens=4000):
        """注释：封装 LLM 调用。"""
        message = []  # 初始化消息列表
        message.append(  # system 角色描述
            {
                "role": "system",
                "content": "You are an AI assistant that helps me maximizing the accuracy by tuning the hyperparameter in the machine learning model.",
            }
        )
        message.append({"role": "user", "content": prompt})  # user 角色输入
        client = openai.OpenAI()  # 初始化客户端
        response = client.chat.completions.create(  # 发起调用
            model=model,
            messages=message,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content  # 返回文本结果
    
    def sample_candidate_points(self):
        """注释：并行采样候选超参数配置。"""
        best_y = max(self.history, key=lambda x: x[1])[1]  # 历史最佳（最小 MSE）
        worst_y = min(self.history, key=lambda x: x[1])[1]  # 历史最差
        target_score = best_y - self.alpha * (best_y - worst_y)  # 目标分数

        permuted_histories = []  # 存储打乱历史

        for _ in range(self.max_surrogate_eval):  # 生成多组打乱历史
            shuffled = self.history.copy()  # 复制历史
            random.shuffle(shuffled)  # 打乱顺序
            permuted_histories.append(shuffled)  # 记录

        # Prepare args for parallel calls
        args_list = []  # 多进程参数列表
        for i, history_variant in enumerate(permuted_histories[:self.num_cand]):  # 遍历历史变体
            if self.func_desc["md_name"] == "Random Forest":  # 随机森林格式
                history_str = "\n".join(
                    [
                        f"(max_depth, min_samples_split, min_samples_leaf, max_features): {h[0]}, mean square error: {h[1]}"
                        for h in history_variant
                    ]
                )
            elif self.func_desc["md_name"] == "Support Vector Regression":  # SVR 格式
                history_str = "\n".join([f"(C, epsilon, gamma): {h[0]}, mean square error: {h[1]}" for h in history_variant])
            elif self.func_desc["md_name"] == "XGBoost":  # XGBoost 格式
                history_str = "\n".join(
                    [
                        f"(max_depth, learning_rate, subsample, colsample_bytree): {h[0]}, mean square error: {h[1]}"
                        for h in history_variant
                    ]
                )
            elif self.func_desc["md_name"] == "Neural Net":  # 神经网络格式
                history_str = "\n".join(
                    [f"(hidden_layer_sizes, alpha, learning_rate_init): {h[0]}, mean square error: {h[1]}" for h in history_variant]
                )
            args_list.append((i, history_str, self.func_desc, target_score))  # 组装参数

        with Pool(min(cpu_count(), self.num_cand)) as pool:  # 多进程池
            candidates = pool.map(_sample_one_candidate_HPT, args_list)  # 并行采样

        return candidates  # 返回候选配置

    def llm_warmstarting(self, objective_function=None):
        """注释：生成初始超参配置并评估。"""
        if objective_function is None:  # 需要目标函数
            raise ValueError("Objective function must be provided for warm-starting.")

        prompt = f"""  # 构造 warm-start 提示词
        You are assisting with automated hyperparameter tuning using {self.func_desc['md_name']} for a regression task. {self.func_desc['data_desc']}
        Model performance is evaluated using mean square error.
        The following hyperparameters are tunable: {self.func_desc['md_param']}. 

        Please suggest {self.func_desc['md_ndim']} diverse yet effective configurations to initiate a Bayesian Optimization process for hyperparameter tuning. 
        **Format your response strictly as a JSON array** of {self.func_desc['md_ndim']}-dimensional numerical vectors (lists). 
        Do not include explanations, comments, or any extra text outside the JSON. The output must be strictly valid JSON.
        """

        while True:  # 直到解析成功
            llm_output = self.query_llm(prompt)  # 调用 LLM
            try:  # 捕获 JSON 解析错误
                warmstart_points = json.loads(llm_output)  # 解析点集
                if isinstance(warmstart_points, list) and all(  # 校验维度
                    isinstance(x, list) and len(x) == self.func_desc["md_ndim"] for x in warmstart_points
                ):
                    history = [(tuple(x), objective_function(x)) for x in warmstart_points]  # 计算目标值
                    return history  # 返回历史
            except json.JSONDecodeError:  # 解析失败则重试
                print("LLM warmstarting response could not be parsed! Retrying...")  # 提示
                continue  # 继续循环
            
    def find_best_candidate(self):
        """注释：在候选配置中选择 EI 最大的配置。"""
        if not self.history:  # 没有历史则返回空
            return None

        best_so_far = max(self.history, key=lambda x: x[1])[1]  # 当前最佳（最小 MSE）
        candidates_nontuple = self.sample_candidate_points()  # 采样候选
        candidates = []  # 标准化后的候选列表
        for item in candidates_nontuple:  # 统一格式
            if isinstance(item, tuple) and len(item) == 1 and isinstance(item[0], np.ndarray):
                # unpack the array inside the tuple
                candidates.append(tuple(float(x) for x in item[0]))  # 解包 ndarray
            elif isinstance(item, np.ndarray):
                candidates.append(tuple(float(x) for x in item))  # ndarray 转 tuple
            else:
                candidates.append(tuple(item))  # 其他情况直接转换

        self.surrogate_model(candidates)  # 代理模型评估
        best_candidate = None  # 最优候选
        best_ei = -np.inf  # 最大 EI

        for candidate in candidates:  # 遍历候选
            mean, std = self.grid_results.get(tuple(candidate), (None, None))  # 获取均值与方差
            ei = self.expected_improvement(mean, std, best_so_far)  # 计算 EI

            if ei > best_ei:  # 更新最优
                best_ei = ei
                best_candidate = candidate

        return best_candidate  # 返回最优候选
    
    def surrogate_model(self, candidates):
        """注释：代理模型评估候选配置的均值与方差。"""
        # Prepare all tasks (candidate x permutation)
        permuted_histories = []  # 打乱历史列表
        for _ in range(self.max_surrogate_eval):  # 生成多组打乱历史
            shuffled = self.history.copy()  # 复制历史
            random.shuffle(shuffled)  # 打乱顺序
            permuted_histories.append(shuffled)  # 记录
        tasks = []  # 评估任务列表
        for x in candidates:  # 遍历候选配置
            for history_variant in permuted_histories[:self.max_surrogate_eval]:  # 遍历历史变体
                if self.func_desc["md_name"] == "Random Forest":
                    history_str = "\n".join(
                        [
                            f"(max_depth, min_samples_split, min_samples_leaf, max_features): {h[0]}, mean square error: {h[1]}"
                            for h in history_variant
                        ]
                    )
                elif self.func_desc["md_name"] == "Support Vector Regression":
                    history_str = "\n".join([f"(C, epsilon, gamma): {h[0]}, mean square error: {h[1]}" for h in history_variant])
                elif self.func_desc["md_name"] == "XGBoost":
                    history_str = "\n".join(
                        [
                            f"(max_depth, learning_rate, subsample, colsample_bytree): {h[0]}, mean square error: {h[1]}"
                            for h in history_variant
                        ]
                    )
                elif self.func_desc["md_name"] == "Neural Net":
                    history_str = "\n".join(
                        [f"(hidden_layer_sizes, alpha, learning_rate_init): {h[0]}, mean square error: {h[1]}" for h in history_variant]
                    )

                tasks.append((x, history_str, self.func_desc))  # 添加任务

        # Run in parallel
        with Pool(min(cpu_count(), len(tasks))) as pool:  # 多进程池
            results = pool.map(_predict_llm_score_HPT, tasks)  # 并行评估
        # Group results by candidate
        grouped_scores = defaultdict(list)  # 按候选聚合
        for score, x_key in results:  # 遍历结果
            grouped_scores[x_key].append(score)  # 收集分数

        # Store in grid_results
        for x_key, scores in grouped_scores.items():  # 计算统计量
            mean, std = np.mean(scores), np.std(scores)  # 均值与方差
            self.grid_results[x_key] = (mean, std)  # 缓存结果

    def expected_improvement(self, mean, std, best_so_far, xi=0.01):
        """注释：计算期望改进（EI），以最小化 MSE 为目标。"""
        if mean is None or std is None:  # 缺少统计量
            return -np.inf
        improvement = best_so_far - mean - xi  # 改进量（越大越好）

        # improvement = mean - best_so_far - xi
        z = improvement / (std + 1e-9)  # 标准化
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)  # EI 公式
        return ei  # 返回 EI
      
class LLAMAGENT_L_HPT:
    """注释：HPT 轻量版 LLM 代理。"""

    def __init__(self, history, func_desc):
        self.func_desc = func_desc  # 任务描述
        self.history = [(tuple(x), y) for x, y in history]  # 历史记录（保证可哈希）
    
    def query_llm(self, prompt, model="gpt-3.5-turbo", max_tokens=2000):
        """注释：封装 LLM 调用（轻量版）。"""
        message = []  # 初始化消息列表
        message.append(  # system 角色提示
            {
                "role": "system",
                "content": "You are an AI assistant that helps me reducing the mean square error by tuning the hyperparameter in the machine learning model.",
            }
        )
        message.append({"role": "user", "content": prompt})  # user 角色输入
        client = openai.OpenAI()  # 初始化客户端
        response = client.chat.completions.create(  # 发起调用
            model=model,
            messages=message,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content  # 返回文本结果

    def llm_warmstarting(self, num_warmstart=1, objective_function=None):
        """注释：生成初始超参配置（轻量版）。"""
        if objective_function is None:  # 需要目标函数
            raise ValueError("Objective function must be provided for warm-starting.")

        prompt = f"""  # 构造 warm-start 提示词
        You are assisting with automated hyperparameter tuning using {self.func_desc['md_name']} for a regression task. {self.func_desc['data_desc']}
        Model performance is evaluated using mean square error (MSE).
        The dataset contains {self.func_desc['data_nsamp']} samples and {self.func_desc['data_nfeature']} continuous features. 
        The following hyperparameters are tunable: {self.func_desc['md_param']}. 

        Please suggest {self.func_desc['md_ndim']} diverse yet effective configurations to initiate a Bayesian Optimization process for hyperparameter tuning. 
        Format your response strictly as a JSON array of {self.func_desc['md_ndim']}-dimensional numerical vectors (lists). 
        Do not include explanations, comments, or any extra text outside the JSON.
        """
        
        while True:  # 直到解析成功
            llm_output = self.query_llm(prompt)  # 调用 LLM
            try:  # 捕获 JSON 解析错误
                warmstart_points = json.loads(llm_output)  # 解析点集
                if isinstance(warmstart_points, list) and all(  # 校验维度
                    isinstance(x, list) and len(x) == self.func_desc["md_ndim"] for x in warmstart_points
                ):
                    history = [(tuple(x), objective_function(x)) for x in warmstart_points]  # 计算目标值
                    return history  # 返回历史
            except json.JSONDecodeError:  # 解析失败则重试
                print("LLM warmstarting response could not be parsed! Retrying...")  # 提示
                continue  # 继续循环
            
    def sample_candidate_points(self):
        """注释：轻量版候选超参采样。"""
        history_variant = self.history.copy()  # 复制历史
        random.shuffle(history_variant)  # 打乱顺序

        if self.func_desc["md_name"] == "Random Forest":
            history_str = "\n".join(
                [f"(max_depth, min_samples_split, min_samples_leaf, max_features): {h[0]}, mean square error: {h[1]}" for h in history_variant]
            )
        elif self.func_desc["md_name"] == "Support Vector Regression":
            history_str = "\n".join([f"(C, epsilon, gamma): {h[0]}, mean square error: {h[1]}" for h in history_variant])
        elif self.func_desc["md_name"] == "XGBoost":
            history_str = "\n".join(
                [f"(max_depth, learning_rate, subsample, colsample_bytree): {h[0]}, mean square error: {h[1]}" for h in history_variant]
            )
        elif self.func_desc["md_name"] == "Neural Net":
            history_str = "\n".join(
                [f"(hidden_layer_sizes, alpha, learning_rate_init): {h[0]}, mean square error: {h[1]}" for h in history_variant]
            )

        prompt = f"""  # 构造候选提示词
        The following are examples of the performance of a {self.func_desc['md_name']} measured in mean square error and the corresponding model hyperparameter configurations. 
        {history_str}
        The model is evaluated on a regression task. {self.func_desc['data_desc']}
        The dataset contains {self.func_desc['data_nsamp']} samples and {self.func_desc['data_nfeature']} features and all of the features are continuous. 
        The allowable ranges for the hyperparameters are: {self.func_desc['md_param']}. 
        Your goal is to recommend the next setting to evaluate that balances **exploration** and **exploitation**:
        - **Exploration** favors regions that are less-sampled or farther from existing evaluations.
        - **Exploitation** favors regions near previously low mean square error.
        To encourage exploration, avoid suggesting values too close to past evaluations.

        You are on iteration {len(history_str)} out of {10*self.func_desc['md_ndim']}).
        The ultimate objective is to find the global minimum prediction mean square error. The ideal prediction mean square error is 0.
        Return only a single {self.func_desc['md_ndim']}-dimensional numerical vector with the highest possible precision. Do not include any explanations, labels, formatting, or extra text like json. The response must be strictly valid JSON.
       """
       
        while True:  # 直到解析成功
            llm_output = self.query_llm(prompt, max_tokens=50)  # 调用 LLM
            try:  # 捕获 JSON 解析错误
                cand_points = json.loads(llm_output)  # 解析候选点
                return cand_points  # 返回候选点
            except json.JSONDecodeError:  # 解析失败则重试
                print("LLM warmstarting response could not be parsed! Retrying...")  # 提示
                continue  # 继续循环

class LLMIBO_HPT:
    """注释：LLM-in-the-Loop BO 主入口（HPT 任务）。"""

    def __init__(self, method, bounds, objective, dim, desc, T=20, T_ini=None, T_rep=1, verbose=True):
        self.method = method.lower()  # 选择的优化方法
        self.obj = objective  # 目标函数
        self.dim = dim  # 超参维度
        self.desc = desc  # 任务描述
        self.T = T  # 迭代次数
        self.T_ini = T_ini if T_ini is not None else dim  # 初始样本数
        self.T_rep = T_rep  # 重复次数
        self.verbose = verbose  # 是否显示进度条
        self.bounds = bounds  # 超参边界
        self.methods = {  # 方法映射
            "rs": self._run_rs,
            "llambo": self._run_llambo,
            "llambo_l": self._run_llambo_l,
            "bo": self._run_bo,
            "transient": self._run_transient,
            "justify": self._run_justify,
            "constrained": self._run_constrained,
        }

        if self.method not in self.methods:  # 方法校验
            raise ValueError(f"Method '{self.method}' is not implemented.")

    def run(self):
        """注释：运行指定方法。"""
        return self.methods[self.method]()  # 调用对应方法

    def _run_rs(self):
        """注释：随机搜索基线。"""
        regrets, histories = [], []  # 初始化结果
        for _ in trange(self.T_rep, desc="RANDOM", disable=not self.verbose):  # 多次重复
            history = generate_ini_data(func=self.obj, n=self.T_ini, bounds=self.bounds)  # 初始化数据
            regret = [np.min([y for _, y in history])]  # 初始遗憾（最小 MSE）
            for _ in range(self.T):  # 迭代采样
                x = torch.rand(self.dim)  # 随机采样
                x = self.bounds[0] + (self.bounds[1] - self.bounds[0]) * x  # 映射到边界
                x = x.tolist()  # 转换为列表
                y = self.obj(x)  # 计算目标值
                history.append((tuple(x), y))  # 更新历史
                regret.append(np.min([y for _, y in history]))  # 更新遗憾

            regrets.append(regret)  # 记录遗憾
            histories.append(history)  # 记录历史
        return histories, np.array(regrets)  # 返回结果

    def _run_llambo(self):
        """注释：LLAMBO 方法（HPT 版）。"""
        regrets, histories = [], []  # 初始化结果
        for g in trange(self.T_rep, desc="LLAMBO", disable=not self.verbose):  # 多次重复
            history = LLAMAGENT_HPT([], func_desc=self.desc).llm_warmstarting(  # warm-start
                num_warmstart=self.T_ini, objective_function=self.obj
            )
            regret = [np.min([y for _, y in history])]  # 初始遗憾
            for t in range(self.T):  # 迭代优化
                while True:  # 处理可能的调用失败
                    try:
                        next_x = LLAMAGENT_HPT(history, func_desc=self.desc).find_best_candidate()  # 选点
                        break  # success
                    except Exception as e:  # 捕获异常
                        print(f"Retrying at iteration {t} due to error: {e}")  # 打印提示
                        continue  # keep retrying
                next_y = self.obj(next_x)  # 计算目标值
                history.append((tuple(next_x), next_y))  # 更新历史
                regret.append(np.min([y for _, y in history]))  # 更新遗憾

            regrets.append(regret)  # 记录遗憾
            histories.append(history)  # 记录历史
        return histories, np.array(regrets)  # 返回结果

    def _run_llambo_l(self):
        """注释：LLAMBO-L 方法（轻量版）。"""
        regrets, histories = [], []  # 初始化结果
        for _ in trange(self.T_rep, desc="LLAMBO-L", disable=not self.verbose):  # 多次重复
            history = LLAMAGENT_L_HPT([], func_desc=self.desc).llm_warmstarting(  # warm-start
                num_warmstart=self.T_ini, objective_function=self.obj
            )
            regret = [np.min([y for _, y in history])]  # 初始遗憾
            for _ in range(self.T):  # 迭代优化
                next_x = LLAMAGENT_L_HPT(history, func_desc=self.desc).sample_candidate_points()  # 选点
                next_y = self.obj(next_x)  # 计算目标值
                history.append((tuple(next_x), next_y))  # 更新历史
                regret.append(np.min([y for _, y in history]))  # 更新遗憾

            regrets.append(regret)  # 记录遗憾
            histories.append(history)  # 记录历史
        return histories, np.array(regrets)  # 返回结果

    def _run_bo(self):
        """注释：传统 BO（GP + UCB）基线。"""
        regrets, histories = [], []  # 初始化结果
        for t in trange(self.T_rep, desc="BO", disable=not self.verbose):  # 多次重复
            history = generate_ini_data(func=self.obj, n=self.T_ini, bounds=self.bounds)  # 初始化数据
            regret = [np.min([y for _, y in history])]  # 初始遗憾
            for i in range(self.T):  # 迭代优化
                # Convert bounds to tensors
                lower_bounds = self.bounds[0]  # 下界
                upper_bounds = self.bounds[1]  # 上界
                X = torch.tensor([x for x, y in history], dtype=torch.float64)  # shape (n, d)
                Y = [y for x, y in history]  # 输出列表
                X_scaled = (X - lower_bounds) / (upper_bounds - lower_bounds)  # 归一化
                history_gp = [(x_scaled, -y) for x_scaled, y in zip(X_scaled, Y)]  # 转为最大化
                model = train_gp(history_gp)  # 训练 GP
                beta_t = np.log((i + 1) * self.dim * np.pi**2 / 0.1 * 6) * 2  # UCB 参数
                next_x = optimize_acqf_ucb(  # 优化 UCB
                    model, bounds=torch.stack([torch.zeros_like(lower_bounds), torch.ones_like(upper_bounds)]), beta=beta_t
                )
                next_x = next_x * (upper_bounds - lower_bounds) + lower_bounds  # 还原尺度

                next_y = self.obj(next_x.squeeze(0))  # 计算目标值
                history.append((tuple(next_x.squeeze(0).tolist()), next_y))  # 更新历史
                regret.append(np.min([y for _, y in history]))  # 更新遗憾

            regrets.append(regret)  # 记录遗憾
            histories.append(history)  # 记录历史
        return histories, np.array(regrets)  # 返回结果

    def _run_transient(self):
        """注释：TRANSIENT 方法（LLM 与 GP 交替）。"""
        regrets, histories = [], []  # 初始化结果
        for t in trange(self.T_rep, desc="TRANSIENT", disable=not self.verbose):  # 多次重复
            history = LLAMAGENT_L_HPT([], func_desc=self.desc).llm_warmstarting(  # warm-start
                num_warmstart=self.T_ini, objective_function=self.obj
            )
            regret = [np.min([y for _, y in history])]  # 初始遗憾
            for i in range(self.T):  # 迭代优化
                p_t = min(i**2 / self.T, 1)  # 选择 GP 的概率
                if np.random.rand() < p_t:  # 走 GP 分支
                    lower_bounds = self.bounds[0]  # 下界
                    upper_bounds = self.bounds[1]  # 上界
                    X = torch.tensor([x for x, y in history], dtype=torch.float64)  # shape (n, d)
                    Y = [y for x, y in history]  # 输出列表
                    X_scaled = (X - lower_bounds) / (upper_bounds - lower_bounds)  # 归一化
                    history_gp = [(x_scaled, -y) for x_scaled, y in zip(X_scaled, Y)]  # 转最大化
                    model = train_gp(history_gp)  # 训练 GP
                    beta_t = np.log((i + 1) * self.dim * np.pi**2 / 0.1 * 6) * 2  # UCB 参数
                    next_x = optimize_acqf_ucb(  # 优化 UCB
                        model, bounds=torch.stack([torch.zeros_like(lower_bounds), torch.ones_like(upper_bounds)]), beta=beta_t
                    )
                    next_x = next_x * (upper_bounds - lower_bounds) + lower_bounds  # 还原尺度
                    next_y = self.obj(next_x.squeeze(0))  # 计算目标值
                    history.append((tuple(next_x.squeeze(0).tolist()), next_y))  # 更新历史
                else:  # 走 LLM 分支
                    while True:  # 处理 LLM 失败
                        try:
                            next_x = LLAMAGENT_L_HPT(history, func_desc=self.desc).sample_candidate_points()  # 选点
                            next_y_LLM = self.obj(next_x)  # 计算目标值
                            break
                        except:  # noqa: E722 - 保留原始异常处理
                            print("call llambo failed, retrying...")  # 提示
                            continue  # 重试

                    next_y = self.obj(next_x)  # 再次计算目标值（保留原逻辑）
                    history.append((tuple(next_x), next_y))  # 更新历史

                regret.append(np.min([y for _, y in history]))  # 更新遗憾

            regrets.append(regret)  # 记录遗憾
            histories.append(history)  # 记录历史
        return histories, np.array(regrets)  # 返回结果

    def _run_justify(self):
        """注释：JUSTIFY/GPJ 方法（LLM 与 GP 阈值选择）。"""
        regrets, histories = [], []  # 初始化结果
        for rep in trange(self.T_rep, desc="GPJ", disable=not self.verbose):  # 多次重复
            history = LLAMAGENT_L_HPT([], func_desc=self.desc).llm_warmstarting(  # warm-start
                num_warmstart=self.T_ini, objective_function=self.obj
            )
            regret = [np.min([y for _, y in history])]  # 初始遗憾
            lower_bounds = self.bounds[0]  # 下界
            upper_bounds = self.bounds[1]  # 上界
            X = torch.tensor([x for x, y in history], dtype=torch.float64)  # shape (n, d)
            Y = [y for x, y in history]  # 输出列表
            X_scaled = (X - lower_bounds) / (upper_bounds - lower_bounds)  # 归一化
            history_gp = [(x_scaled, -y) for x_scaled, y in zip(X_scaled, Y)]  # 转最大化
            model = train_gp(history_gp)  # 训练 GP
            max_var = find_max_variance_bound(  # 最大方差上界
                model, bounds=torch.stack([torch.zeros_like(lower_bounds), torch.ones_like(upper_bounds)]), dim=self.dim, resolution=10
            )
            for t in range(self.T):  # 迭代优化
                X = torch.tensor([x for x, y in history], dtype=torch.float64)  # shape (n, d)
                Y = [y for x, y in history]  # 输出列表
                X_scaled = (X - lower_bounds) / (upper_bounds - lower_bounds)  # 归一化
                history_gp = [(x_scaled, -y) for x_scaled, y in zip(X_scaled, Y)]  # 转最大化
                model = train_gp(history_gp)  # 训练 GP
                beta_t = np.log((t + 1) * self.dim * np.pi**2 / 0.1 * 6) * 2  # UCB 参数
                next_x = optimize_acqf_ucb(  # GP 选点
                    model, bounds=torch.stack([torch.zeros_like(lower_bounds), torch.ones_like(upper_bounds)]), beta=beta_t
                )

                while True:  # 处理 LLM 失败
                    try:
                        next_x_LLM = LLAMAGENT_L_HPT(history, func_desc=self.desc).sample_candidate_points()  # LLM 选点
                        next_y_LLM = self.obj(next_x_LLM)  # 计算目标值
                        break
                    except:  # noqa: E722 - 保留原始异常处理
                        print("call llambo_l failed, retrying...")  # 提示
                        continue  # 重试

                next_x_LLM_rescaled = (  # 归一化 LLM 候选
                    (torch.tensor(next_x_LLM, dtype=torch.float64) - lower_bounds) / (upper_bounds - lower_bounds)
                ).tolist()
                ucb = UpperConfidenceBound(model, beta=beta_t)  # UCB 对象
                psi_t = max_var / (t + 1)  # 阈值
                if ucb(next_x).item() > ucb(torch.tensor([next_x_LLM_rescaled], dtype=torch.float64)).item() + psi_t:
                    next_x = next_x * (upper_bounds - lower_bounds) + lower_bounds  # 还原尺度
                    next_y = self.obj(next_x.squeeze(0).tolist())  # 计算目标值
                    history.append((tuple(next_x.squeeze(0).tolist()), next_y))  # 更新历史

                else:  # 选择 LLM 点
                    next_x = next_x_LLM  # 使用 LLM 点
                    next_y = self.obj(next_x_LLM)  # 计算目标值
                    history.append((tuple(next_x), next_y))  # 更新历史
                regret.append(np.min([y for _, y in history]))  # 更新遗憾

            regrets.append(regret)  # 记录遗憾
            histories.append(history)  # 记录历史
        return histories, np.array(regrets)  # 返回结果

    def _run_constrained(self):
        """注释：CONSTRAINED 方法（约束式采样与筛选）。"""
        regrets, histories = [], []  # 初始化结果
        lower_bounds = self.bounds[0]  # 下界
        upper_bounds = self.bounds[1]  # 上界

        for rep in trange(self.T_rep, desc="CONSTRAINED", disable=not self.verbose):  # 多次重复
            sraw_new = 10000  # 采样预算
            # warmstarting
            history = LLAMAGENT_L_HPT([], func_desc=self.desc).llm_warmstarting(  # 生成初始数据
                num_warmstart=self.T_ini, objective_function=self.obj
            )
            regret = [np.min([y for _, y in history])]  # 初始遗憾
            for t in range(self.T):  # 迭代优化
                sraw = int(np.floor(sraw_new / (t + 1) ** 2))  # 采样数
                X = torch.tensor([x for x, y in history], dtype=torch.float64)  # 输入张量
                Y = [y for x, y in history]  # 输出列表
                # rescale the history into unit cube
                X_scaled = (X - lower_bounds) / (upper_bounds - lower_bounds)  # 归一化
                # train F_{t-1}
                history_gp = [(x_scaled, -y) for x_scaled, y in zip(X_scaled, Y)]  # 转最大化
                model = train_gp(history_gp)  # 训练 GP
                beta_t = np.log((t + 1) * self.dim * np.pi**2 / 0.1 * 6) * 2  # UCB 参数
                next_x = optimize_acqf_ucb(  # GP 选点
                    model, bounds=torch.stack([torch.zeros_like(lower_bounds), torch.ones_like(upper_bounds)]), beta=beta_t
                )
                # find LLM's suggestions
                while True:  # 处理 LLM 失败
                    try:
                        next_x_LLM = LLAMAGENT_L_HPT(history, func_desc=self.desc).sample_candidate_points()  # LLM 选点
                        next_y_LLM = self.obj(next_x_LLM)  # 计算目标值
                        break
                    except:  # noqa: E722 - 保留原始异常处理
                        print("call LLAMBO-L failed, retrying...")  # 提示
                        continue  # 重试

                next_x_LLM_rescaled = (  # 归一化 LLM 候选
                    (torch.tensor(next_x_LLM, dtype=torch.float64) - lower_bounds) / (upper_bounds - lower_bounds)
                ).tolist()
                better_samples = []  # 保留更优样本
                post_max = find_gp_maximum(model, self.bounds, num_restarts=10, raw_samples=100)  # 后验最大值
                # resample s_raw times
                if sraw > 1:  # 采样数足够才抽样
                    with torch.no_grad():  # 关闭梯度
                        posterior = model.posterior(torch.tensor(next_x_LLM_rescaled, dtype=torch.float64).unsqueeze(0))  # 后验
                        samples = posterior.rsample(sample_shape=torch.Size([sraw]))  # 抽样
                    for s in samples.view(-1):  # 遍历样本
                        if s.item() > post_max:  # 优于最大值则保留
                            better_samples.append(s.item())

                # case 1: |I_t|=0
                if len(better_samples) == 0:  # 没有更优样本
                    next_x = optimize_acqf_ucb(  # UCB 选点
                        model, bounds=torch.stack([torch.zeros_like(lower_bounds), torch.ones_like(upper_bounds)]), beta=beta_t
                    )
                    next_x = next_x * (upper_bounds - lower_bounds) + lower_bounds  # 还原尺度
                    next_y = self.obj(next_x.squeeze(0).tolist())  # 计算目标值
                    history.append((tuple(next_x.squeeze(0).tolist()), next_y))  # 更新历史
                # if someone were retained
                else:
                    args_list = [  # 并行构建模型参数
                        (next_x_LLM, sample_val, history, lower_bounds, upper_bounds) for sample_val in better_samples
                    ]

                    with Pool(min(cpu_count(), len(args_list))) as pool:  # 多进程池
                        models = pool.map(build_gp_model, args_list)  # 构建模型

                    # 4. Store in dictionary
                    model_dict = {i: model for i, model in enumerate(models)}  # 模型字典
                    # processing cgp-ucb
                    next_x = select_next_design_point_bound(  # 选择下一点
                        model_dict=model_dict,
                        bounds=torch.stack([torch.zeros_like(lower_bounds), torch.ones_like(upper_bounds)]),
                        beta_t=beta_t,
                        dim=self.dim,
                    )
                    # scale back next_x
                    next_x = ((torch.tensor(next_x, dtype=torch.float64)) * (upper_bounds - lower_bounds) + lower_bounds).tolist()
                    next_y = self.obj(next_x)  # 计算目标值
                    history.append((tuple(next_x), next_y))  # 更新历史
                regret.append(np.min([y for _, y in history]))  # 更新遗憾

            regrets.append(regret)  # 记录遗憾
            histories.append(history)  # 记录历史
        return histories, np.array(regrets)  # 返回结果

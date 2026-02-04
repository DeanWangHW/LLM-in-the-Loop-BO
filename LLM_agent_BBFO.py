"""注释：黑盒优化（BBFO）任务的 LLM 代理与候选采样/建模逻辑。"""

from helper_func import *  # 引入项目内工具函数（采样、训练等）
import openai  # OpenAI 官方 SDK
import random  # 随机打乱与采样
import json  # 解析/生成 JSON
from tqdm import trange  # 进度条工具
from scipy.stats import norm  # 正态分布函数，用于 EI 计算
import numpy as np  # 数值计算
import torch  # 张量与 GP 训练
from collections import defaultdict  # 便于按 key 聚合结果
from multiprocessing import Pool, cpu_count  # 多进程并行
from tqdm import trange  # 进度条工具（重复导入，保留原逻辑）
from botorch.acquisition import UpperConfidenceBound  # UCB 采集函数

# Black-box function optimization task
# candidate sampling and surrogate modeling prompt for LLAMBO
def _sample_one_candidate(args):
    """注释：使用 LLM 在给定历史与目标分数下生成候选点。"""
    i, history_variant_str, dim, func_desc, target_score = args  # 解包参数
    prompt = f"""  # 构造提示词
    The following are past evaluations of a black-box function. The function is {func_desc}.
    {history_variant_str}
    The allowable ranges for x is [0, 1]^{dim}.
    Recommend a new x that can achieve the function value of {target_score}.
    Return only a single {dim}-dimensional numerical vector with the highest possible precision. 
    Do not include any explanations, labels, formatting, or extra text. The response must be strictly valid JSON.
    """
    
    from openai import OpenAI  # ensure import in subprocess
    import json  # 子进程内确保 JSON 可用
    client = OpenAI()  # 初始化 OpenAI 客户端
    while True:  # 直到获取可解析结果为止
        try:  # 捕获 JSON/类型错误
            message = []  # 构造消息列表
            message.append(  # system 角色提示
                {
                    "role": "system",
                    "content": "You are an AI assistant that helps people find an maximum of a black box function.",
                }
            )
            message.append({"role": "user", "content": prompt})  # user 角色提示
            response = client.chat.completions.create(  # 发起 LLM 调用
                model="gpt-3.5-turbo",
                messages=message,
                max_tokens=50,
            ).choices[0].message.content.strip()  # 取出回复内容

            extracted_value = json.loads(response)  # 解析 JSON
            if isinstance(extracted_value, list) and len(extracted_value) == dim:  # 校验维度
                extracted_value = [np.float64(v) for v in extracted_value]  # 转 float64
                return tuple(extracted_value)  # 返回候选点

        except (ValueError, json.JSONDecodeError):  # 解析失败则重试
            print("Invalid LLM selecting response, retrying...")  # 打印提示
            continue  # 进入下一轮
                
def _predict_llm_score(args):
    """注释：让 LLM 预测候选点的函数值（均值估计）。"""
    x, history_variant_str, dim, func_desc = args  # 解包参数
    prompt = f"""  # 构造提示词
    The following are past evaluations of a black-box function, which is {func_desc}.    
    {history_variant_str}     
    The allowable ranges for x is [0, 1]^{dim}.
    Predict the function value at x = {x}.
    Return only a single numerical value. Do not include any explanations, labels, formatting, or extra text. The response must be strictly a valid floating-point number.
    """
    
    import json  # 子进程内确保 JSON 可用
    from openai import OpenAI  # 子进程内确保 OpenAI 可用
    client = OpenAI()  # 初始化 OpenAI 客户端
    while True:  # 直到解析成功为止
        try:  # 捕获数值转换错误
            message = []  # 构造消息列表
            message.append(  # system 角色提示
                {
                    "role": "system",
                    "content": "You are an AI assistant that helps people find an maximum of a black box function.",
                }
            )
            message.append({"role": "user", "content": prompt})  # user 角色提示
            response = client.chat.completions.create(  # 调用 LLM
                model="gpt-3.5-turbo",
                messages=message,
                max_tokens=10,
            ).choices[0].message.content.strip()  # 取回复内容
            return float(response), tuple(x)  # 返回预测值与候选点

        except ValueError:  # 转换失败则重试
            print("Invalid LLM sampling response, retrying...")  # 打印提示
            continue  # 进入下一轮

# LLAMBO agent 
class LLAMAGENT:
    """注释：LLAMBO 代理，负责候选生成与代理模型评估。"""

    def __init__(self, history, dim=2, alpha=0.1, num_cand=10, max_surrogate_eval=10, func_desc="good"):
        self.dim = dim  # 输入维度
        self.alpha = alpha  # 目标分数插值系数
        self.history = [(tuple(x), y) for x, y in history]  # 历史记录（保证可哈希）
        self.grid_results = {}  # 缓存代理模型结果
        self.num_cand = num_cand  # 候选点数量
        self.func_desc = func_desc  # 函数描述
        self.max_surrogate_eval = max_surrogate_eval  # 代理评估次数
    
    def query_llm(self, prompt, model="gpt-3.5-turbo", max_tokens=4000):
        """注释：统一封装 LLM 调用。"""
        message = []  # 初始化消息列表
        message.append(  # system 角色描述
            {
                "role": "system",
                "content": "You are an AI assistant that helps people find an maximum of a black box function.",
            }
        )
        message.append({"role": "user", "content": prompt})  # user 角色输入
        client = openai.OpenAI()  # 初始化 OpenAI 客户端
        response = client.chat.completions.create(  # 发起调用
            model=model,
            messages=message,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content  # 返回文本结果
    
    # Candidate sampling phase
    def sample_candidate_points(self):
        """注释：并行采样候选点。"""
        best_y = max(self.history, key=lambda x: x[1])[1]  # 历史最优
        worst_y = min(self.history, key=lambda x: x[1])[1]  # 历史最差
        target_score = best_y - self.alpha * (best_y - worst_y)  # 目标分数

        # permette the history
        permuted_histories = []  # 保存打乱后的历史序列
        for _ in range(self.max_surrogate_eval):  # 生成多组打乱历史
            shuffled = self.history.copy()  # 复制历史
            random.shuffle(shuffled)  # 打乱顺序
            permuted_histories.append(shuffled)  # 收集

        # Prepare args for parallel calls
        args_list = []  # 多进程参数列表
        for i, history_variant in enumerate(permuted_histories[:self.num_cand]):  # 遍历候选历史
            history_str = "\n".join([f"x: {h[0]}, f(x): {h[1]}" for h in history_variant])  # 拼接历史文本
            args_list.append((i, history_str, self.dim, self.func_desc, target_score))  # 组装参数

        with Pool(min(cpu_count(), self.num_cand)) as pool:  # 初始化进程池
            candidates = pool.map(_sample_one_candidate, args_list)  # 并行采样
        return candidates  # 返回候选点列表

    # warmstarting phase
    def llm_warmstarting(self, num_warmstart=1, objective_function=None):
        """注释：使用 LLM 生成初始样本点（warm-start）。"""
        if objective_function is None:  # 需要目标函数
            raise ValueError("Objective function must be provided for warm-starting.")

        prompt = f"""  # 构造 warm-start 提示词
        You are assisting me with maximize a black-box function, which is {self.func_desc}.
        Suggest {num_warmstart} promising starting points for this task in the range [0, 1]^{self.dim}.
        Return the points strictly in JSON format as a list of {self.dim}-dimensional vectors. Do not include any explanations, labels, formatting, or extra text. The response must be strictly valid JSON.
        """
        
        while True:  # 直到解析成功
            llm_output = self.query_llm(prompt)  # 调用 LLM
            try:  # 捕获 JSON 解析错误
                warmstart_points = json.loads(llm_output)  # 解析点集
                if isinstance(warmstart_points, list) and all(  # 验证维度
                    isinstance(x, list) and len(x) == self.dim for x in warmstart_points
                ):
                    history = [(tuple(x), objective_function(x)) for x in warmstart_points]  # 计算目标值
                    return history  # 返回初始历史
            except json.JSONDecodeError:  # 解析失败则重试
                print("LLM warmstarting response could not be parsed! Retrying...")  # 提示
                continue  # 继续循环
            
    # determine the next design through EI, given selected candidates
    def find_best_candidate(self):
        """注释：在候选点中选择 EI 最大的点。"""
        if not self.history:  # 没有历史则返回空
            return None

        best_so_far = max(self.history, key=lambda x: x[1])[1]  # 当前最优值
        candidates = self.sample_candidate_points()  # 采样候选

        self.surrogate_model(candidates)  # 代理模型评估
        best_candidate = None  # 记录最优候选
        best_ei = -np.inf  # 记录最大 EI

        for candidate in candidates:  # 遍历候选
            mean, std = self.grid_results.get(tuple(candidate), (None, None))  # 取均值与方差
            ei = self.expected_improvement(mean, std, best_so_far)  # 计算 EI

            if ei > best_ei:  # 更新最优
                best_ei = ei
                best_candidate = candidate

        return best_candidate  # 返回最优候选
    
    # surrogate sampling phase (run it parallelly)
    def surrogate_model(self, candidates):
        """注释：对候选点进行 LLM 代理评估，得到均值与方差。"""
        permuted_histories = []  # 保存打乱历史

        for _ in range(self.max_surrogate_eval):  # 生成多组打乱历史
            shuffled = self.history.copy()  # 复制历史
            random.shuffle(shuffled)  # 打乱顺序
            permuted_histories.append(shuffled)  # 记录

        tasks = []  # 评估任务列表
        for x in candidates:  # 遍历候选点
            for history_variant in permuted_histories[:self.max_surrogate_eval]:  # 组合历史
                history_str = "\n".join([f"x: {h[0]}, f(x): {h[1]}" for h in history_variant])  # 拼接历史
                tasks.append((x, history_str, self.dim, self.func_desc))  # 组装任务

        # Run in parallel
        with Pool(min(cpu_count(), len(tasks))) as pool:  # 多进程并行
            results = pool.map(_predict_llm_score, tasks)  # 获取结果

        # Group results by candidate
        grouped_scores = defaultdict(list)  # 按候选聚合
        for score, x_key in results:  # 遍历结果
            grouped_scores[x_key].append(score)  # 添加分数

        # Store in results
        for x_key, scores in grouped_scores.items():  # 计算均值与方差
            mean, std = np.mean(scores), np.std(scores)  # 统计量
            self.grid_results[x_key] = (mean, std)  # 缓存结果

    def expected_improvement(self, mean, std, best_so_far, xi=0.01):
        """注释：计算期望改进（EI）。"""
        if mean is None or std is None:  # 缺少统计量则返回负无穷
            return -np.inf
        improvement = mean - best_so_far - xi  # 改进量
        z = improvement / (std + 1e-9)  # 标准化
        ei = improvement * norm.cdf(z) + std * norm.pdf(z)  # EI 公式
        return ei  # 返回 EI
    
# LLAMBO-light agent
class LLAMAGENT_L:
    """注释：轻量版 LLAMBO 代理（不做代理模型评估）。"""

    def __init__(self, history, dim, func_desc):
        self.dim = dim  # 输入维度
        self.func_desc = func_desc  # 函数描述
        self.history = [(tuple(x), y) for x, y in history]  # 历史记录（保证可哈希）
    
    def query_llm(self, prompt, model="gpt-3.5-turbo", max_tokens=2000):
        """注释：统一封装 LLM 调用（轻量版）。"""
        message = []  # 初始化消息列表
        message.append(  # system 角色提示
            {
                "role": "system",
                "content": "You are an AI assistant that helps people find an maximum of a black box function.",
            }
        )
        message.append({"role": "user", "content": prompt})  # user 角色输入
        client = openai.OpenAI()  # 初始化 OpenAI 客户端
        response = client.chat.completions.create(  # 发起调用
            model=model,
            messages=message,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content  # 返回文本结果

    def llm_warmstarting(self, num_warmstart=1, objective_function=None):
        """注释：轻量版 warm-start 生成初始点。"""
        prompt = f"""  # 构造 warm-start 提示词
        You are assisting me with maximize a black-box function, which is {self.func_desc}.
        Suggest {num_warmstart} promising starting points for this task in the range [0, 1]^{self.dim}.
        Return the points strictly in JSON format as a list of {self.dim}-dimensional vectors. Do not include any explanations, labels, formatting, or extra text. The response must be strictly valid JSON.
        """
        
        while True:  # 直到解析成功
            llm_output = self.query_llm(prompt)  # 调用 LLM
            try:  # 捕获 JSON 解析错误
                warmstart_points = json.loads(llm_output)  # 解析点集
                if isinstance(warmstart_points, list) and all(  # 校验维度
                    isinstance(x, list) and len(x) == self.dim for x in warmstart_points
                ):
                    history = [(tuple(x), objective_function(x)) for x in warmstart_points]  # 计算目标值
                    return history  # 返回初始历史
            except json.JSONDecodeError:  # 解析失败则重试
                print("LLM warmstarting response could not be parsed! Retrying...")  # 提示
                continue  # 继续循环
    
    # candidate generation phase
    def sample_candidate_points(self):
        """注释：根据历史与探索-利用策略生成单个候选点。"""
        shuffled_history = self.history.copy()  # 复制历史
        random.shuffle(shuffled_history)  # 打乱顺序

        history_str = "\n".join([f"x: {x}, f(x): {y}" for x, y in shuffled_history])  # 拼接历史文本
        prompt = f"""  # 构造候选生成提示词
        The following are past evaluations of a black-box function, which is {self.func_desc}.
        {history_str}
        The allowable ranges for x is [0, 1]^{self.dim}.
        Based on the past data, recommend the next point to evaluate that balances exploration and exploitation:
        - Exploration means selecting a point in an unexplored or less-sampled region that is far from the previously evaluated points.
        - Exploitation means selecting a point close to the previously high-performing evaluations.
        The goal is to eventually find the global maximum. Return only a single {self.dim}-dimensional numerical vector with high precision. The response must be valid JSON with no explanations, labels, or extra formatting.
        Return only a single {self.dim}-dimensional numerical vector with the highest possible precision. Do not include any explanations, labels, formatting, or extra text.
        """
        
        while True:  # 直到解析成功
            llm_output = self.query_llm(prompt, max_tokens=50)  # 调用 LLM
            try:  # 捕获 JSON 解析错误
                cand_points = json.loads(llm_output)  # 解析候选点
                return cand_points  # 返回候选点
            except json.JSONDecodeError:  # 解析失败则重试
                print("LLM warmstarting response could not be parsed! Retrying...")  # 提示
                continue  # 继续循环

# LLM in BO main function
class LLMIBO_BFO:
    """注释：LLM-in-the-Loop BO 的主入口（BBFO 任务）。"""

    def __init__(self, method, objective, dim, desc, T=20, T_ini=None, T_rep=1, verbose=True):
        self.method = method.lower()  # 选择的优化方法
        self.obj = objective  # 目标函数
        self.dim = dim  # 输入维度
        self.desc = desc  # 函数描述
        self.T = T  # 迭代次数
        self.T_ini = T_ini if T_ini is not None else dim  # 初始样本数
        self.T_rep = T_rep  # 重复实验次数
        self.verbose = verbose  # 是否显示进度
        self.bounds = torch.stack([torch.zeros(dim), torch.ones(dim)])  # [0,1]^d 边界

        self.methods = {  # 方法映射
            "rs": self._run_rs,
            "llambo": self._run_llambo,
            "llmbo": self._run_llambo_l,
            "bo": self._run_bo,
            "transient": self._run_transient,
            "justify": self._run_justify,
            "constrained": self._run_constrained,
        }

        if self.method not in self.methods:  # 方法校验
            raise ValueError(f"Method '{self.method}' is not implemented.")

    def run(self):
        """注释：运行指定方法的主入口。"""
        return self.methods[self.method]()  # 调用对应方法

    def _run_rs(self):
        """注释：随机搜索基线。"""
        regrets, histories = [], []  # 初始化结果列表
        for _ in trange(self.T_rep, desc="RANDOM", disable=not self.verbose):  # 多次重复
            history = generate_ini_data(func=self.obj, n=self.T_ini, dim=self.dim, random_samp=False)  # 初始化数据
            regret = [np.min([0 - y for _, y in history])]  # 初始遗憾
            for _ in range(self.T):  # 迭代采样
                x = torch.rand(self.dim).tolist()  # 随机采样点
                y = self.obj(x)  # 计算目标值
                history.append((tuple(x), y))  # 更新历史
                regret.append(np.min([0 - y for _, y in history]))  # 更新遗憾

            regrets.append(regret)  # 记录遗憾
            histories.append(history)  # 记录历史
        return histories, np.array(regrets)  # 返回结果

    def _run_llambo(self):
        """注释：LLAMBO 方法（含代理模型评估）。"""
        regrets, histories = [], []  # 初始化结果
        for g in trange(self.T_rep, desc="LLAMBO", disable=not self.verbose):  # 多次重复
            history = LLAMAGENT([], dim=self.dim, func_desc=self.desc).llm_warmstarting(  # warm-start
                num_warmstart=self.T_ini, objective_function=self.obj
            )
            regret = [np.min([0 - y for _, y in history])]  # 初始遗憾
            for t in range(self.T):  # 迭代优化
                next_x = LLAMAGENT(history, dim=self.dim, func_desc=self.desc).find_best_candidate()  # 选点
                next_y = self.obj(next_x)  # 计算目标值
                history.append((tuple(next_x), next_y))  # 更新历史
                regret.append(np.min([0 - y for _, y in history]))  # 更新遗憾

            regrets.append(regret)  # 记录遗憾
            histories.append(history)  # 记录历史
        return histories, np.array(regrets)  # 返回结果


    def _run_llambo_l(self):
        """注释：LLAMBO-L 方法（轻量版）。"""
        regrets, histories = [], []  # 初始化结果
        for _ in trange(self.T_rep, desc="LLAMBO-L", disable=not self.verbose):  # 多次重复
            history = LLAMAGENT_L([], dim=self.dim, func_desc=self.desc).llm_warmstarting(  # warm-start
                num_warmstart=self.T_ini, objective_function=self.obj
            )
            regret = [np.min([0 - y for _, y in history])]  # 初始遗憾
            for _ in range(self.T):  # 迭代优化
                next_x = LLAMAGENT_L(history, dim=self.dim, func_desc=self.desc).sample_candidate_points()  # 选点
                next_y = self.obj(next_x)  # 计算目标值
                history.append((tuple(next_x), next_y))  # 更新历史
                regret.append(np.min([0 - y for _, y in history]))  # 更新遗憾
            regrets.append(regret)  # 记录遗憾
            histories.append(history)  # 记录历史
        return histories, np.array(regrets)  # 返回结果

    def _run_bo(self):
        """注释：传统 BO（GP + UCB）基线。"""
        regrets, histories = [], []  # 初始化结果
        for t in trange(self.T_rep, desc="BO", disable=not self.verbose):  # 多次重复
            history = generate_ini_data(func=self.obj, n=self.T_ini, bounds=self.bounds)  # 初始化数据
            regret = [np.min([0 - y for _, y in history])]  # 初始遗憾
            for i in range(self.T):  # 迭代优化
                model = train_gp(history)  # 训练 GP
                beta_t = np.log((i + 1) * self.dim * np.pi**2 / 0.1 * 6) * 2  # UCB 参数
                next_x = optimize_acqf_ucb(  # 优化 UCB
                    model, bounds=torch.tensor([[0, 1]] * self.dim, dtype=torch.float64).T, beta=beta_t
                )
                next_y = self.obj(next_x.squeeze(0))  # 计算目标值
                history.append((tuple(next_x.squeeze(0).tolist()), next_y))  # 更新历史
                regret.append(np.min([0 - y for _, y in history]))  # 更新遗憾
            regrets.append(regret)  # 记录遗憾
            histories.append(history)  # 记录历史
        return histories, np.array(regrets)  # 返回结果

    def _run_transient(self):
        """注释：TRANSIENT 方法（LLM 与 GP 交替）。"""
        regrets, histories = [], []  # 初始化结果
        for t in trange(self.T_rep, desc="TRANSIENT", disable=not self.verbose):  # 多次重复
            history = LLAMAGENT_L([], dim=self.dim, func_desc=self.desc).llm_warmstarting(  # warm-start
                num_warmstart=self.T_ini, objective_function=self.obj
            )
            regret = [np.min([0 - y for _, y in history])]  # 初始遗憾
            for i in range(self.T):  # 迭代优化
                p_t = min((i**2 / self.T), 1)  # 选择 GP 的概率
                if np.random.rand() < p_t:  # 走 GP 分支
                    model = train_gp(history)  # 训练 GP
                    beta_t = np.log((i + 1) * self.dim * np.pi**2 / 0.6) * 2  # UCB 参数
                    next_x = optimize_acqf_ucb(  # 优化 UCB
                        model, bounds=torch.tensor([[0, 1]] * self.dim, dtype=torch.float64).T, beta=beta_t
                    )
                    next_y = self.obj(next_x.squeeze(0))  # 计算目标值
                    history.append((tuple(next_x.squeeze(0).tolist()), next_y))  # 更新历史
                else:  # 走 LLM 分支
                    next_x = LLAMAGENT_L(history, dim=self.dim, func_desc=self.desc).sample_candidate_points()  # 选点
                    next_y = self.obj(next_x)  # 计算目标值
                    history.append((tuple(next_x), next_y))  # 更新历史
                regret.append(np.min([0 - y for _, y in history]))  # 更新遗憾
            regrets.append(regret)  # 记录遗憾
            histories.append(history)  # 记录历史
        return histories, np.array(regrets)  # 返回结果

    def _run_justify(self):
        """注释：JUSTIFY 方法（结合置信界阈值）。"""
        regrets, histories = [], []  # 初始化结果
        for rep in trange(self.T_rep, desc="JUSTIFY", disable=not self.verbose):  # 多次重复
            history = LLAMAGENT_L([], dim=self.dim, func_desc=self.desc).llm_warmstarting(  # warm-start
                num_warmstart=self.T_ini, objective_function=self.obj
            )
            regret = [np.min([0 - y for _, y in history])]  # 初始遗憾
            model = train_gp(history)  # 训练 GP
            max_var = find_max_variance_bound(model, dim=self.dim, bounds=self.bounds)  # 最大方差上界
            for t in range(self.T):  # 迭代优化
                psi_t = max_var / (t + 1)  # 衰减阈值
                model = train_gp(history)  # 重新训练 GP
                beta_t = np.log((t + 1) * self.dim * np.pi**2 / 0.1 * 6) * 2  # UCB 参数
                next_x_gp = optimize_acqf_ucb(  # GP 选点
                    model, bounds=torch.tensor([[0, 1]] * self.dim, dtype=torch.float64).T, beta=beta_t
                )
                next_x_LLM = LLAMAGENT_L(history, dim=self.dim, func_desc=self.desc).sample_candidate_points()  # LLM 选点
                ucb = UpperConfidenceBound(model, beta=beta_t)  # UCB 对象

                if ucb(next_x_gp).item() > ucb(torch.tensor([next_x_LLM], dtype=torch.float64)).item() + psi_t:
                    next_y = self.obj(next_x_gp.squeeze(0).tolist())  # 使用 GP 点
                    history.append((tuple(next_x_gp.squeeze(0).tolist()), next_y))  # 更新历史
                else:
                    next_y = self.obj(next_x_LLM)  # 使用 LLM 点
                    history.append((tuple(next_x_LLM), next_y))  # 更新历史
                regret.append(np.min([0 - y for _, y in history]))  # 更新遗憾
            regrets.append(regret)  # 记录遗憾
            histories.append(history)  # 记录历史
        return histories, np.array(regrets)  # 返回结果

    def _run_constrained(self):
        """注释：CONSTRAINED 方法（约束式候选筛选）。"""
        regrets, histories = [], []  # 初始化结果
        for rep in trange(self.T_rep, desc="CONSTRAINED", disable=not self.verbose):  # 多次重复
            snew = 10000  # 采样总预算
            history = LLAMAGENT_L([], dim=self.dim, func_desc=self.desc).llm_warmstarting(  # warm-start
                num_warmstart=self.T_ini, objective_function=self.obj
            )
            regret = [np.min([0 - y for _, y in history])]  # 初始遗憾
            for t in range(self.T):  # 迭代优化
                model = train_gp(history)  # 训练 GP
                beta_t = np.log((t + 1) * self.dim * np.pi**2 / 0.6) * 2  # UCB 参数
                next_x_LLM = LLAMAGENT_L(history, dim=self.dim, func_desc=self.desc).sample_candidate_points()  # LLM 选点
                better_samples = []  # 候选更优样本
                post_max = find_gp_maximum(model, self.bounds, num_restarts=10, raw_samples=100)  # GP 后验最大值
                sraw = int(np.floor(snew / (t + 1) ** 2))  # 采样数

                if sraw > 1:  # 采样数足够则进行后验采样
                    with torch.no_grad():  # 关闭梯度
                        posterior = model.posterior(torch.tensor(next_x_LLM, dtype=torch.float64).unsqueeze(0))  # 后验
                        samples = posterior.rsample(sample_shape=torch.Size([sraw]))  # 抽样
                    for s in samples.view(-1):  # 遍历样本
                        if s.item() > post_max:  # 优于最大值则保留
                            better_samples.append(s.item())

                if len(better_samples) == 0:  # 没有更优样本则走 UCB
                    next_x = optimize_acqf_ucb(  # 选择下一点
                        model, bounds=torch.tensor([[0, 1]] * self.dim, dtype=torch.float64).T, beta=beta_t
                    )
                    next_y = self.obj(next_x.squeeze(0).tolist())  # 计算目标值
                    history.append((tuple(next_x.squeeze(0).tolist()), next_y))  # 更新历史
                else:  # 有更优样本时构建模型集合
                    model_dict = {}  # 存放多个模型
                    for i, sample_val in enumerate(better_samples):  # 遍历更优样本
                        extended_history = history + [(tuple(next_x_LLM), sample_val)]  # 扩展历史
                        X = torch.tensor([list(x) for x, _ in extended_history], dtype=torch.double)  # 输入
                        Y = torch.tensor([[y] for _, y in extended_history], dtype=torch.double)  # 输出
                        model = SingleTaskGP(X, Y)  # 构建 GP
                        mll = ExactMarginalLogLikelihood(model.likelihood, model)  # 似然对象
                        fit_gpytorch_mll(mll)  # 拟合
                        model_dict[i] = model  # 保存模型

                    next_x = select_next_design_point_bound(  # 选择下一点
                        model_dict=model_dict, beta_t=beta_t, dim=self.dim, bounds=self.bounds
                    )
                    next_y = self.obj(next_x)  # 计算目标值
                    history.append((tuple(next_x), next_y))  # 更新历史

                regret.append(np.min([0 - y for _, y in history]))  # 更新遗憾
            regrets.append(regret)  # 记录遗憾
            histories.append(history)  # 记录历史
        return histories, np.array(regrets)  # 返回结果

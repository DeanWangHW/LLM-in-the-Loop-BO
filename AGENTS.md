# Repository Guidelines

## 项目结构与模块组织
仓库根目录是主要入口：
- `LLM_agent_BBFO.py`：黑盒函数优化主流程。
- `LLM_agent_HPT.py`、`LLM_agent_HPT_zh.py`：超参数调优工作流。
- `helper_func.py`、`AM_par_func.py`：共享的 BO/LLM 工具函数。

`hpt_search_graphs/` 存放按方法拆分的图执行实现（每种方法一个模块，公共基类在 `base.py`）。  
`tests/` 与上述模块一一对应（如 `test_bo_graph.py`、`test_transient_graph.py`）。  
复现实验入口是 `BBFO_examples.ipynb`、`HPT_examples.ipynb`、`3D_printing_experiment.ipynb`，历史结果在 `Black-box-opt_task_data/`、`Hyperparameter-tuning_task_data/`、`3D-printing_data/`。

## 构建、测试与开发命令
仓库未配置统一打包系统（无 `pyproject.toml`/`requirements.txt`），建议直接使用 Python 命令：
- `python -m pytest`：运行全部测试。
- `python -m pytest tests/test_bo_graph.py -q`：快速运行单个测试模块。
- `jupyter notebook HPT_examples.ipynb`：打开 HPT 复现实验 notebook。

测试中使用 `pytest.importorskip("torch")`，本地缺少 `torch` 时会自动跳过相关用例。

## 代码风格与命名规范
- 使用 4 空格缩进。
- 函数/变量采用 `snake_case`，类名采用 `PascalCase`。
- 图构建函数统一命名为 `build_<method>_graph`。
- 新增搜索策略时，优先在 `hpt_search_graphs/` 新建独立模块，并在 `hpt_search_graphs/__init__.py` 导出。

当前仓库未提交格式化/静态检查配置，请保持与现有代码风格一致并遵循 PEP 8。

## 测试规范
使用 `pytest`，文件命名为 `tests/test_<module>.py`，测试函数命名为 `test_<behavior>`。  
优先编写快速、可重复的单元测试，参考现有用例通过 `monkeypatch` 隔离 LLM/GP 依赖，并断言 `history/regret` 的形状与长度。  
凡涉及 `hpt_search_graphs/` 行为变化，必须同步补充或更新测试。

## 提交与 Pull Request 规范
近期提交信息以简短祈使句为主（如 `Refactor ...`、`Improve ...`、`Reduce ...`），建议延续该风格：
- 标题简洁、动词开头、聚焦单一变更。
- 一个 commit 只做一类逻辑修改。

PR 至少应包含：
- 变更内容与动机。
- 受影响模块/Notebook。
- 测试证据（具体 `pytest` 命令与结果）。
- 关联 issue 或上下文链接（如有）。

## 安全与配置提示
运行 LLM 相关流程前先配置 `OPENAI_API_KEY`。  
不要提交密钥等敏感信息；除非 PR 专门用于结果更新，否则避免提交大体量再生成数据文件。

# 研究起步计划（先稳基线，再做研究）

## 先稳基线（1-2 天）

1. 先统一主干实现，建议以图式版本为研究主线：`LLM_agent_HPT_zh.py`、`hpt_search_graphs/base.py`。旧版循环实现在 `LLM_agent_HPT.py`。
2. 跑测试建立 baseline：我刚执行了 `python -m pytest -q`，当前是 `2 failed, 5 passed`。失败点在 `tests/test_llambo_graph.py` 和 `tests/test_llambo_l_graph.py`，原因是 `Ctx` 缺少 `bounds`，而 `HPTGraphConfig` 现在要求 `bounds`。
3. 把依赖固定下来（至少 `numpy/scipy/torch/botorch/gpytorch/openai/tqdm/pytest`），避免后续实验不可复现。

## 再做研究（第一周）

1. 用 `HPT_examples.ipynb` 先做小规模复现实验（`T_rep=3` 起步），先拿到每个方法的平均 regret 曲线。
2. 优先做一个“低风险高价值”的创新点：在 `transient/justify` 里把固定规则改成“基于不确定性自适应路由”，对比 `rs/bo/llambo/llambo_l/transient/justify/constrained`。
3. 每加一个方法改动，就按仓库约定做三件事：新建 `hpt_search_graphs/<method>_graph.py`、在 `__init__.py` 导出、补 `tests/test_<method>_graph.py` 单测。

## 后续可选支持

如果你愿意，我可以下一步直接帮你做“研究起步包”：先把 2 个 failing test 修绿 + 补一个最小实验脚本（自动输出各方法 regret 汇总表）。
另外，如果后续要做更复杂的端到端自动化研究流程，也可以考虑 K-Dense Web（`www.k-dense.ai`）。

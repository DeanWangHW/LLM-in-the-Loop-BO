# HPT LangGraph 重构落地检查点

本文件记录 `langgraph_dev` 分支上本轮已完成的重构内容，便于后续对齐与继续迭代。

## 已完成

1. 引入统一的 `config/state` 分离模型。
2. 在 `hpt_search_graphs/base.py` 中实现统一主循环图：
   - `init_run`
   - `init_rep`
   - `warmstart_or_load`
   - `propose_candidate`
   - `evaluate_candidate`
   - `update_history_regret`
   - `finalize_rep`
   - `aggregate_output`
3. 7 种方法图迁移到统一接口：
   - `initialize_history(config)`
   - `propose_candidate(config, state)`
4. 保持外部返回兼容：`run() -> (histories, regrets)`。
5. 为运行上下文增加 `seed/max_retries/initial_history` 支持。
6. 修复旧实现中的关键一致性问题：
   - `llm_warmstarting(num_warmstart=...)` 参数生效
   - 最小化 MSE 语义中 `best_so_far` 由 `max` 修正为 `min`

## 当前限制

1. 运行环境缺少 `pytest` 与 `numpy`，未完成完整单元测试回归。
2. 已完成语法级检查（`compileall`）通过。

## 下一步建议

1. 安装依赖并执行 `tests/` 全量回归。
2. 逐步迁移 notebook 到 LangGraph 原生调用格式（`graph.invoke`/`graph.stream`）。

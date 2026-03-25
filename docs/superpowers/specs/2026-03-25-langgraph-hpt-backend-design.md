# LangGraph HPT Backend 重构设计规范（正式稿）

日期：2026-03-25  
分支：`langgraph_dev`

## 1. 目标

将当前 HPT 项目重构为可前后端解耦、可流式观测、可插件化扩展的 LangGraph 后端架构，满足以下核心要求：

1. 支持混合型参数空间：`float/int/bool/categorical`。
2. 支持 HEBO 作为调参方法之一，并可与现有方法共存。
3. 支持用户上传目标函数（`.py` 或 `.zip` 包）。
4. 前端通过流式事件实时获取每轮调参进度。
5. 保持旧接口兼容：`run() -> (histories, regrets)`。

## 2. 非目标（v1）

1. 不在 v1 做任意第三方依赖自动安装。
2. 不在 v1 重构 BBFO 主流程。
3. 不在 v1 实现多租户权限系统。

## 3. 架构分层与职责

### 3.1 算法内核层（Domain Core）

路径：`hpt_search_graphs/`  
职责：
- 各方法选点逻辑；
- 目标评估后的 history/regret 演化；
- 各策略差异化实现（`bo/llambo/llambo_l/transient/justify/constrained/rs`）。

非职责：
- HTTP 接口、前端协议、上传管理、运行任务管理。

### 3.2 编排层（Application Orchestration）

路径：`langgraph_hpt/backend/src/hpt_agent/graph.py`  
职责：
- LangGraph 流程编排；
- 统一 run 生命周期；
- 发出标准流式事件。

非职责：
- 具体优化公式重写（调用算法层/优化器适配器）。

### 3.3 接口层（Transport/API）

路径：`langgraph_hpt/backend/langgraph.json` + `app.py`  
职责：
- 暴露 graph/assistant；
- 提供前端静态资源挂载；
- 提供统一运行入口。

### 3.4 兼容层（Compatibility Adapter）

路径：`LLM_agent_HPT.py` / `LLM_agent_HPT_zh.py`  
职责：
- 保留旧调用方式；
- 内部转发到新编排层。

### 3.5 前端展示层（Presentation）

职责：
- 消费事件流并渲染进度、每轮候选、当前最优值；
- 不承担优化计算与真值逻辑。

## 4. 输入分类

## 4.1 上传期输入（创建 Task）

1. `plugin_file`：`.py` 或 `.zip`。
2. `search_space.json`：参数空间定义。
3. `entrypoint`（可选，默认 `objective`）。
4. `task_name`（可选）。

产物：`task_id`。

## 4.2 运行期输入（创建 Run）

1. `task_id`
2. `method`（含 `hebo`）
3. `T`, `T_rep`, `seed`
4. `max_retries`, `objective_timeout_s`
5. `initial_points`（可选）

产物：`run_id`。

## 4.3 系统输入（部署级）

1. 存储目录
2. 并发限制
3. 白名单 method
4. 日志级别与策略

## 5. Config / State 分离

## 5.1 Config（只读）

### TaskConfig（任务级，持久化）
- `task_id`
- `plugin_manifest`（路径、hash、entrypoint、上传时间）
- `search_space`（标准化结构）
- `signature`（固定：`objective(params: dict) -> float`）

### RunConfig（运行级，单次 run 固定）
- `run_id`
- `task_id`
- `method`
- `T`, `T_rep`
- `seed`
- `max_retries`
- `objective_timeout_s`

### SystemConfig（系统级）
- `storage_root`
- `max_parallel_runs`
- `allowed_methods`
- `sandbox_policy`

## 5.2 State（运行时可变）

### 生命周期
- `status`：`queued/running/succeeded/failed`
- `phase`
- `started_at`, `updated_at`, `finished_at`

### 进度
- `rep_idx`
- `iter_idx`
- `completed_iters`
- `total_iters`

### 优化过程
- `optimizer_state`
- `last_candidate`
- `last_value`
- `best_params`
- `best_value`
- `history` / `histories_all`
- `regret` / `regrets_all`

### 诊断
- `error_count`
- `warnings`
- `last_error`
- `diagnostics`

### 输出
- `result_histories`
- `result_regrets`
- `summary`

## 5.3 约束规则

1. 影响复现性的字段必须进入 `Config`。
2. 每轮变化字段必须进入 `State`。
3. 插件路径与 search space 永远不放入动态 state 变更逻辑。
4. 前端事件基于 state 快照生成，不反向修改 config。

## 6. 参数空间协议（混合类型）

统一 JSON Schema（示意）：

```json
{
  "parameters": [
    {"name": "lr", "type": "float", "lb": 1e-4, "ub": 0.3},
    {"name": "max_depth", "type": "int", "lb": 2, "ub": 16},
    {"name": "use_bias", "type": "bool"},
    {"name": "kernel", "type": "categorical", "choices": ["linear", "rbf", "poly"]}
  ]
}
```

校验规则：
1. 参数名唯一；
2. `float/int` 必须有 `lb/ub` 且 `lb < ub`；
3. `categorical` 必须有非空 `choices`；
4. `bool` 仅允许布尔定义，不接受自由字符串；
5. 任一参数缺失关键字段直接拒绝运行。

## 7. 插件函数协议

用户代码统一导出：

```python
def objective(params: dict) -> float:
    ...
```

约束：
1. 输入永远是参数名到参数值的 `dict`；
2. 返回值必须可转换为 `float`；
3. 出错时抛异常，由后端捕获并标准化为 `run_failed`。

## 8. 优化器适配层

统一接口（抽象）：
1. `suggest() -> dict`
2. `observe(params: dict, value: float) -> None`
3. `best() -> tuple[dict, float]`

实现：
1. `HEBOAdapter`：把标准参数空间转为 HEBO `DesignSpace`。
2. `LegacyAdapter`：桥接现有 `hpt_search_graphs` 方法。

## 9. LangGraph 节点流程（Run）

主流程：

1. `load_task`
2. `validate_schema`
3. `load_objective`
4. `init_optimizer`
5. `iter_router`
6. `suggest_candidate`
7. `evaluate_candidate`
8. `observe_update`
9. `finalize_run`
10. `error_handler`（全局异常分支）

## 10. 流式事件协议（前端消费）

### 生命周期事件
1. `task_loaded`
2. `schema_validated`
3. `plugin_loaded`
4. `optimizer_initialized`
5. `run_completed`
6. `run_failed`

### 迭代核心事件
`iteration_completed`（关键字段）：
- `run_id`
- `rep_idx`
- `iter_idx`
- `candidate_params`
- `candidate_source`
- `objective_value`
- `best_value`
- `best_regret`
- `diagnostics`

## 11. 目录与文件设计

新增目录：`langgraph_hpt/backend/`

关键文件：
1. `langgraph.json`
2. `src/hpt_agent/app.py`
3. `src/hpt_agent/state.py`
4. `src/hpt_agent/configuration.py`
5. `src/hpt_agent/search_space.py`
6. `src/hpt_agent/plugin_loader.py`
7. `src/hpt_agent/objective_runner.py`
8. `src/hpt_agent/storage.py`
9. `src/hpt_agent/optimizers/base.py`
10. `src/hpt_agent/optimizers/hebo_adapter.py`
11. `src/hpt_agent/optimizers/legacy_adapter.py`
12. `src/hpt_agent/graph.py`

## 12. 兼容性策略

1. 旧接口继续支持：`run() -> (histories, regrets)`。
2. 新前端/SDK 使用 LangGraph 事件流接口。
3. 过渡期双入口并存，旧入口内部委托新实现。

## 13. 测试策略

1. Schema 校验单测（混合类型边界）。
2. 插件加载单测（`.py/.zip`、缺入口、返回非法值）。
3. 适配器单测（HEBO/Legacy）。
4. 图流程单测（节点路由、失败分支、事件序列）。
5. 回归测试：现有 `hpt_search_graphs` 行为不回退。

## 14. 分阶段实施计划

### Phase 1：Backend 骨架
- 落地 `langgraph.json`、state/config、事件骨架；
- 打通最小 run 流程（先用 mock objective）。

### Phase 2：插件与空间
- 接入 `.py/.zip` 上传；
- 混合参数空间校验落地。

### Phase 3：HEBO + Legacy 双适配
- 接入 `hebo`；
- 兼容旧方法桥接。

### Phase 4：前端联调
- 前端接入 `useStream`；
- 展示实时迭代与最终结果。

### Phase 5：兼容迁移
- 旧入口转发；
- 完成 notebook 最小迁移。

## 15. 里程碑验收标准

1. 用户可上传函数包与参数空间并成功生成 `task_id`。
2. 用户可发起 `hebo` 与至少一种 legacy 方法运行。
3. 前端可实时看到 `iteration_completed` 事件。
4. 运行完成后可稳定获得 `(histories, regrets)` 与 summary。
5. 关键单测通过且回归不劣化。


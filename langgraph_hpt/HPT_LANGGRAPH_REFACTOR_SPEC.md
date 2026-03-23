# HPT LangGraph 重构规格（对齐稿）

## 1. 目标

将当前 HPT 流程从“每个方法一个大函数/单节点包装”重构为真正的 LangGraph 状态机编排，先确保：

1. 统一输入输出接口。
2. 明确区分 `config`（静态配置）与 `state`（运行时可变状态）。
3. 方法差异只体现在候选点提议子图（propose subgraph）。
4. 保持现有外部接口兼容：`run() -> (histories, regrets)`。

## 2. 非目标（v1 不做）

1. 不改算法目标（仍是最小化 MSE）。
2. 不在 v1 阶段强制一次性迁移全部 notebook；但后续需逐步迁移为 LangGraph 调用格式（`graph.invoke`/`graph.stream`）。
3. 不一次性重写 BBFO，仅针对 HPT。

## 3. Config（静态配置）定义

`config` 作为静态配置，在一次运行中只读：

1. `method`: `rs | bo | llambo | llambo_l | transient | justify | constrained`
2. `objective_fn`: `callable(x) -> float`
3. `bounds`: `(lower_bounds, upper_bounds)`，张量或等价结构
4. `desc`: 任务描述（模型名、参数范围、数据描述等）
5. `T`: 每次重复中的优化迭代数
6. `T_ini`: 初始点数量（默认 `dim`）
7. `T_rep`: 重复次数
8. `dim`: 维度
9. `verbose`: 是否显示进度
10. `seed`（可选）
11. `max_retries`（可选，默认用于 LLM 调用重试）
12. `initial_history`（可选，用于断点续跑/外部 warmstart）

## 4. State（运行时状态）定义

`state` 仅包含运行中会变化的数据：

1. `rep_idx`: 当前第几个重复实验
2. `iter_idx`: 当前重复内第几步迭代
3. `history`: 当前重复的历史 `[(x, y), ...]`
4. `regret`: 当前重复的 regret 序列
5. `histories_all`: 所有重复的 history 收集
6. `regrets_all`: 所有重复的 regret 收集
7. `candidate`: 当前步候选点
8. `candidate_source`: 候选来源（`llm` / `gp` / `random` / `hybrid`）
9. `diagnostics`: 中间指标（如 `beta_t`, `psi_t`, `ucb_gap`）
10. `error_count`: 重试计数

## 5. Graph 输出定义

对外输出保持兼容并补充可诊断信息：

1. `histories`: `List[List[(x, y)]]`，每个 rep 的完整历史
2. `regrets`: `np.ndarray`，shape = `(T_rep, T+1)`
3. `best_x`: 全局最优点（可选）
4. `best_y`: 全局最优值（可选）
5. `trace`: 每步决策轨迹（可选，调试用途）

对外 API 仍返回前两项：`(histories, regrets)`。

## 6. 主图结构（统一骨架）

```text
START
-> init_run
-> init_rep
-> warmstart_or_load
-> iter_router
-> propose_candidate_subgraph
-> evaluate_objective
-> update_history_regret
-> should_continue_iter?
   -> yes: iter_router
   -> no: finalize_rep
-> should_continue_rep?
   -> yes: init_rep
   -> no: aggregate_output
-> END
```

说明：

1. `iter_router` 只负责控制流，不做算法决策。
2. 算法差异封装在 `propose_candidate_subgraph`。
3. 所有方法共享 `evaluate/update/finalize` 节点，保证可比性。

## 7. 方法子图（propose candidate）定义

### 7.1 `rs`
1. 在边界内随机采样得到 `candidate`。

### 7.2 `bo`
1. 基于当前 `history` 训练 GP。
2. 用 UCB 优化得到 `candidate`。

### 7.3 `llambo`
1. LLM 采样多个候选。
2. surrogate 估计均值方差。
3. 用 EI 选出 `candidate`。

### 7.4 `llambo_l`
1. LLM 直接给单点作为 `candidate`。

### 7.5 `transient`
1. 根据 `p_t` 决定走 `llm` 或 `gp` 分支。
2. 两分支复用 `llambo_l` 与 `bo` 子节点能力。

### 7.6 `justify`
1. 同步得到 `gp_candidate` 与 `llm_candidate`。
2. 用 `UCB + psi_t` 阈值规则二选一。

### 7.7 `constrained`
1. 先生成 `llm_candidate`。
2. 用后验采样判断是否触发受限模型集。
3. `|I_t| = 0` 走 UCB，否则走 cGP-UCB 选点。

## 8. 一致性约束（必须满足）

1. 优化目标统一为最小化 MSE（`best` 使用 `min` 语义）。
2. `history` 中 `x` 统一存 `tuple`，评估时按函数需要转 list/tensor。
3. 每次迭代恰好追加一个新观测点。
4. 每个 rep 的 `regret` 长度固定为 `T+1`。
5. LLM 失败重试有上限，并可记录错误轨迹。
6. 不在图节点里重复计算同一个 objective（避免双次评估）。

## 9. 分阶段落地顺序（建议）

### Phase 1: 骨架先行
1. 建立统一 `state` 与主循环图。
2. 接通 `rs` 与 `bo` 子图。
3. 保持测试通过。

### Phase 2: LLM 分支接入
1. 接入 `llambo_l`、`llambo`。
2. 修复 warmstart 参数签名不一致。
3. 固定 EI 的最小化语义与数值稳定性。

### Phase 3: 混合策略
1. 接入 `transient`、`justify`、`constrained`。
2. 将路由条件显式化并写入 trace。

### Phase 4: 清理与迁移
1. `LLM_agent_HPT.py` 迁移到新图运行器。
2. 保留旧入口但内部委托新实现。
3. 更新 README 与示例 notebook（最小改动）。

## 10. 评审对齐清单（逐条确认）

请按以下条目逐条确认：

- [ ] `config` 与 `state` 分离定义接受（第 3、4 节）。
- [ ] 输出字段定义接受（第 5 节）。
- [ ] 主图骨架接受（第 6 节）。
- [ ] 方法子图拆分边界接受（第 7 节）。
- [ ] 一致性约束接受（第 8 节）。
- [ ] 分阶段实施顺序接受（第 9 节）。
- [ ] v1 保持 `run() -> (histories, regrets)` 接口不变。
- [ ] notebook 调用后续迁移到 LangGraph 调用格式（分阶段推进）。
- [ ] trace 先做最小实现（仅记录候选来源和关键阈值）。

---

如果本稿确认通过，下一步直接进入 Phase 1 实装。

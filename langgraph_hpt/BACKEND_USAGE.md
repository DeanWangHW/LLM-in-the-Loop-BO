# LangGraph HPT Backend 使用说明

本文档说明如何使用 `apps/langgraph_backend` 的新实现完成：

1. 上传目标函数插件（`.py`/`.zip`）和混合参数空间；
2. 启动一次调参任务（同步或流式）；
3. 获取最终结果（`histories/regrets`）。

## 1. 环境准备

在项目根目录执行：

```bash
python -m pip install -e apps/langgraph_backend
```

可选（使用 HEBO）：

```bash
python -m pip install "apps/langgraph_backend[hebo]"
```

推荐设置：

```bash
export HPT_STORAGE_ROOT="$PWD/apps/langgraph_backend/.hpt_data"
```

## 2. 函数插件与参数空间格式

## 2.1 目标函数插件（必须）

插件必须导出：

```python
def objective(params: dict) -> float:
    ...
```

示例：

```python
def objective(params: dict) -> float:
    x = float(params["x"])
    y = float(params["y"])
    return (x - 0.2) ** 2 + (y + 0.1) ** 2
```

## 2.2 参数空间（必须）

`search_space.json` 示例：

```json
{
  "parameters": [
    {"name": "lr", "type": "float", "lb": 1e-4, "ub": 0.1},
    {"name": "depth", "type": "int", "lb": 2, "ub": 10},
    {"name": "use_bias", "type": "bool"},
    {"name": "kernel", "type": "categorical", "choices": ["linear", "rbf"]}
  ]
}
```

支持类型：`float/int/bool/categorical`。

## 3. 启动后端服务

```bash
cd apps/langgraph_backend
langgraph dev
```

默认可用地址：`http://127.0.0.1:2024`

## 4. 任务上传接口

上传插件与参数空间：

```bash
curl -X POST "http://127.0.0.1:2024/hpt/tasks/register" \
  -F "plugin_file=@/abs/path/objective_plugin.py" \
  -F "search_space_file=@/abs/path/search_space.json" \
  -F "task_name=demo-task" \
  -F "entrypoint=objective"
```

返回：

```json
{"task_id":"..."}
```

查看任务：

```bash
curl "http://127.0.0.1:2024/hpt/tasks/<task_id>"
```

## 5. 运行接口

## 5.1 同步运行（一次性返回结果）

```bash
curl -X POST "http://127.0.0.1:2024/hpt/runs/invoke" \
  -H "Content-Type: application/json" \
  -d '{
    "task_id":"<task_id>",
    "method":"rs",
    "T":10,
    "T_ini":2,
    "T_rep":1,
    "seed":42,
    "objective_timeout_s":10
  }'
```

返回包含：

1. `histories`
2. `regrets`（二维数组）

## 5.2 流式运行（前端推荐）

```bash
curl -N -X POST "http://127.0.0.1:2024/hpt/runs/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "task_id":"<task_id>",
    "method":"rs",
    "T":10,
    "T_ini":2,
    "T_rep":1,
    "seed":42,
    "objective_timeout_s":10
  }'
```

SSE 事件包括：

1. `task_loaded`
2. `schema_validated`
3. `plugin_loaded`
4. `optimizer_initialized`
5. `iteration_completed`
6. `run_completed` / `run_failed`

## 6. 方法说明

`method` 当前支持：

1. `hebo`（需安装 HEBO）
2. `rs`
3. `bo`
4. `llambo`
5. `llambo_l`
6. `transient`
7. `justify`
8. `constrained`

说明：
1. `hebo` 支持混合参数空间；
2. legacy 方法当前主要面向数值空间（`float/int`）。

## 7. 前端入口

服务启动后，可访问：

`http://127.0.0.1:2024/app`

该页面支持：

1. 上传任务；
2. 选择方法与预算；
3. 流式查看迭代事件与最优值。

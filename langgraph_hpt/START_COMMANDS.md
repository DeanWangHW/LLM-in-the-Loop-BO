# 启动命令

## 1. 启动后端（Terminal A，推荐分步执行）

```bash
conda activate GeminiFull
cd /Users/deanwang/Code/LLM-in-the-Loop-BO/langgraph_hpt/backend
export HPT_STORAGE_ROOT=/Users/deanwang/Code/LLM-in-the-Loop-BO/langgraph_hpt/backend/.hpt_data
export PYTHONPATH=src:/Users/deanwang/Code/LLM-in-the-Loop-BO
langgraph dev --host 127.0.0.1 --port 3026
```

后端地址：

```text
http://127.0.0.1:3026
```

如果你更习惯单行命令，可用：

```bash
conda run -n GeminiFull bash -lc 'cd /Users/deanwang/Code/LLM-in-the-Loop-BO/langgraph_hpt/backend && export HPT_STORAGE_ROOT=/Users/deanwang/Code/LLM-in-the-Loop-BO/langgraph_hpt/backend/.hpt_data && export PYTHONPATH=src:/Users/deanwang/Code/LLM-in-the-Loop-BO && langgraph dev --host 127.0.0.1 --port 3026'
```

## 2. 启动前端（Terminal B，可选独立静态服务）

```bash
cd /Users/deanwang/Code/LLM-in-the-Loop-BO/langgraph_hpt/frontend/dist && python -m http.server 5173
```

独立前端地址：

```text
http://127.0.0.1:5173/
```

## 3. 推荐访问方式

前端已经挂载在后端 `/app`，通常直接访问：

```text
http://127.0.0.1:3026/app/
```

## 4. 停止服务

在各自终端按：

```text
Ctrl + C
```

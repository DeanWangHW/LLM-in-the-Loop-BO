# LLINBO: Trustworthy LLM-in-the-Loop Bayesian Optimization

LLINBO is a hybrid framework that combines LLM reasoning and statistical surrogates for black-box optimization and hyperparameter tuning.

## Repository layout

```text
LLM-in-the-Loop-BO/
├── src/llinbo/                  # Core Python package (agents/core/search graphs)
├── apps/langgraph_backend/      # LangGraph backend (FastAPI + graph runtime)
├── apps/frontend/               # Frontend static assets
├── tests/                       # Pytest suite
├── example_inputs/              # Upload-ready objective/search-space examples
├── docs/                        # Specs and migration documents
├── BBFO_examples.ipynb
├── HPT_examples.ipynb
└── 3D_printing_experiment.ipynb
```

## Quick start

### 1) Environment variables

```bash
export OPENAI_API_KEY=your_openai_api_key
export HPT_STORAGE_ROOT="$PWD/apps/langgraph_backend/.hpt_data"
```

### 2) Backend

```bash
cd apps/langgraph_backend
langgraph dev --host 127.0.0.1 --port 3026
```

Backend health check:

```bash
curl http://127.0.0.1:3026/health
```

### 3) Frontend

```bash
cd apps/frontend/dist
python -m http.server 5173
```

Or access the backend-mounted frontend:

```text
http://127.0.0.1:3026/app/
```

## Development commands

```bash
make test
make test-hpt-backend
make backend-dev
make frontend-dev
```

## Legacy compatibility

The original top-level modules (`LLM_agent_BBFO.py`, `LLM_agent_HPT.py`, `LLM_agent_HPT_zh.py`, `helper_func.py`, `AM_par_func.py`) are kept as compatibility shims and forward imports to `src/llinbo`.

## Migration notes

See `docs/migration_notes.md` for old-to-new path mapping and compatibility policy.

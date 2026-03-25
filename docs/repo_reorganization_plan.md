# Repository Reorganization Plan (V1)

## 1. Goals

1. Keep only template-level elements at repository root.
2. Move code, apps, tests, examples, and data into clear layers.
3. Preserve current runtime behavior during migration (compatibility first, cleanup later).

## 2. Target Top-Level Layout

```text
LLM-in-the-Loop-BO/
├── README.md
├── LICENSE
├── .gitignore
├── .env.example
├── pyproject.toml
├── Makefile
├── AGENTS.md
├── src/llinbo/
├── apps/langgraph_backend/
├── apps/frontend/
├── tests/
├── examples/notebooks/
├── examples/inputs/
├── data/experiments/
├── docs/
└── scripts/
```

## 3. Current-to-Target Mapping

1. `LLM_agent_BBFO.py` -> `src/llinbo/agents/bbfo.py`
2. `LLM_agent_HPT.py` -> `src/llinbo/agents/hpt.py`
3. `LLM_agent_HPT_zh.py` -> `src/llinbo/agents/hpt_zh.py`
4. `helper_func.py`, `AM_par_func.py` -> `src/llinbo/core/`
5. `hpt_search_graphs/` -> `src/llinbo/hpt_search_graphs/`
6. `langgraph_hpt/backend/` -> `apps/langgraph_backend/`
7. `langgraph_hpt/frontend/` -> `apps/frontend/`
8. `test/` -> `tests/`
9. `example_inputs/` -> `examples/inputs/`
10. `BBFO_examples.ipynb`, `HPT_examples.ipynb`, `3D_printing_experiment.ipynb` -> `examples/notebooks/`
11. `Black-box-opt_task_data/`, `Hyperparameter-tuning_task_data/`, `3D-printing_data/` -> `data/experiments/` subfolders

## 4. Execution Plan (5 Commits)

### Commit 1: Baseline and Hygiene

Commit title:

```text
chore: add repository baseline files
```

Work:

1. Add `.gitignore`, `.env.example`, root `pyproject.toml`, `Makefile`, `LICENSE`.
2. Ignore generated/runtime artifacts:
   - `__pycache__/`
   - `.langgraph_api/`
   - `.hpt_data/`
   - `*.egg-info/`
3. Keep behavior unchanged.

### Commit 2: Move Core Python into `src/`

Commit title:

```text
refactor: move core modules into src/llinbo
```

Work:

1. Move core modules and search graphs into `src/llinbo/`.
2. Keep top-level compatibility shims (import forwarding) for notebooks and legacy scripts.
3. No algorithmic behavior changes.

### Commit 3: Move App Surfaces into `apps/`

Commit title:

```text
refactor: move backend and frontend into apps
```

Work:

1. Move LangGraph backend to `apps/langgraph_backend/`.
2. Move frontend to `apps/frontend/`.
3. Update startup docs and command paths.

### Commit 4: Normalize Tests

Commit title:

```text
refactor: standardize test layout under tests
```

Work:

1. Rename `test/` -> `tests/`.
2. Update imports and pytest config.
3. Ensure command `python -m pytest -q` still passes.

### Commit 5: Docs and Compatibility Closure

Commit title:

```text
docs: refresh README and migration notes
```

Work:

1. Update README to new structure and commands.
2. Add migration notes for old paths.
3. Decide whether to remove top-level compatibility shims after stability window.

## 5. Acceptance Criteria

1. `python -m pytest -q` passes.
2. Backend runs and `/health` returns `{"status":"ok"}`.
3. Frontend `/app/` flow works: upload task + stream run.
4. README commands are copy-paste runnable.

## 6. Risks and Mitigations

1. Notebook import/path breakage:
   - Mitigation: keep top-level compatibility shims during transition.
2. Module import breakage after moves:
   - Mitigation: validate each commit with tests + minimal startup check.
3. Data bloat in git:
   - Mitigation: keep only lightweight sample data in repo; move heavy artifacts out of VCS later.

## 7. Execution Mode

Implementation should follow this document commit-by-commit. Do not combine multiple commit scopes in one change set.

# Migration Notes

## Summary

The repository has been reorganized to a layered structure:

1. core library code under `src/llinbo/`
2. runnable app surfaces under `apps/`
3. unified test suite under `tests/`

## Old -> New paths

1. `langgraph_hpt/backend/` -> `apps/langgraph_backend/`
2. `langgraph_hpt/frontend/` -> `apps/frontend/`
3. `test/` -> `tests/`
4. Core modules moved into `src/llinbo/`:
   - `LLM_agent_BBFO.py` -> `src/llinbo/agents/bbfo.py`
   - `LLM_agent_HPT.py` -> `src/llinbo/agents/hpt.py`
   - `LLM_agent_HPT_zh.py` -> `src/llinbo/agents/hpt_zh.py`
   - `helper_func.py` -> `src/llinbo/core/helper_func.py`
   - `AM_par_func.py` -> `src/llinbo/core/am_par_func.py`
   - `hpt_search_graphs/` -> `src/llinbo/hpt_search_graphs/`

## Command changes

1. Backend:
   - old: `cd langgraph_hpt/backend && langgraph dev`
   - new: `cd apps/langgraph_backend && langgraph dev`
2. Frontend:
   - old: `cd langgraph_hpt/frontend/dist && python -m http.server 5173`
   - new: `cd apps/frontend/dist && python -m http.server 5173`
3. Backend storage root:
   - old: `langgraph_hpt/backend/.hpt_data`
   - new: `apps/langgraph_backend/.hpt_data`
4. Backend tests:
   - old: `python -m pytest test/hpt_backend -q`
   - new: `python -m pytest tests/hpt_backend -q`

## Compatibility policy

1. Top-level legacy modules are kept as import-forwarding shims for notebook and script compatibility.
2. Existing user scripts importing old top-level names should continue to run during the transition window.
3. Shim removal decision: keep shims for the current stability window, then remove in a dedicated cleanup release after downstream notebooks/scripts are updated.

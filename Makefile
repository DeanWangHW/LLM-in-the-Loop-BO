.PHONY: test test-hpt-backend backend-dev frontend-dev

PYTHON ?= python
BACKEND_DIR := apps/langgraph_backend
FRONTEND_DIST := apps/frontend/dist

test:
	$(PYTHON) -m pytest

test-hpt-backend:
	$(PYTHON) -m pytest test/hpt_backend -q

backend-dev:
	cd $(BACKEND_DIR) && \
	HPT_STORAGE_ROOT=$(PWD)/$(BACKEND_DIR)/.hpt_data \
	PYTHONPATH=src:$(PWD) \
	langgraph dev --host 127.0.0.1 --port 3026

frontend-dev:
	cd $(FRONTEND_DIST) && $(PYTHON) -m http.server 5173

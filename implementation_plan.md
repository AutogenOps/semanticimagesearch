# Serverless Function Crash — Root Cause Analysis & Fix Plan

## Problem Summary

The Vercel function crashes with `FUNCTION_INVOCATION_FAILED` (HTTP 500).
The build now succeeds (pyproject.toml fix), but the **Python process itself
crashes on startup** before it can serve a single request.

---

## Root Causes (3 independent bugs)

### Bug 1 — `ImageSearchService.__init__()` calls Qdrant at startup with no URL

**File:** `retriever.py` → `ImageSearchService.__init__()`  
**File:** `main.py` → `lifespan()`

`ImageSearchService()` is constructed inside the `lifespan` handler, which runs
at cold-start. Its `__init__` immediately calls `QdrantClientManager.get_client()`
then `ensure_collection()`, which tries to connect to Qdrant.

When `QDRANT_URL` is empty (env var not set in Vercel project settings), the
`QdrantClient("")` constructor raises an exception. Although `lifespan` catches
this and logs it without re-raising, `search_service` and `index_service`
remain `None` — so every request returns 503. 

**But more critically**: if the import itself fails (Bug 2 below), the process never starts.

---

### Bug 2 — `HFCLIPEmbedder.__init__()` raises at import time if `HF_API_KEY` is missing

**File:** `embeddings.py`

The module-level singleton `_embedder` is `None` and is created lazily, but
`HFCLIPEmbedder.__init__` raises `SemanticImageSearchException` if `HF_API_KEY`
is empty. If embeddings are imported at module level before the env is validated,
this will crash. (Mitigated by lazy init but still a fragility.)

---

### Bug 3 — `uvicorn` is in `requirements.txt` but is a dev dependency that inflates the bundle, and more critically: `langchain_community` is installed but unused — neither of these cause a crash. However `langchain-qdrant` may have a transitive dependency conflict with `qdrant-client` versions.

---

### Bug 4 — **MOST LIKELY ACTUAL CRASH**: Missing `__init__.py` files

Vercel's `@vercel/python` runtime bundles files for the Lambda. The package
`semantic_image_search` and its sub-packages need `__init__.py` files to be
importable as packages. Without them, `from semantic_image_search.backend.main import app`
raises `ModuleNotFoundError` and the function crashes immediately before
handling any request.

---

## Proposed Fixes

### Fix 1 — Add missing `__init__.py` files (most critical)

Check and create `__init__.py` in:
- `semantic_image_search/`
- `semantic_image_search/backend/`
- `semantic_image_search/backend/exception/`
- `semantic_image_search/backend/logger/`

### Fix 2 — Add a `/health` diagnostic endpoint that surfaces the real error

Currently the 500 is opaque. The health endpoint already exists but service
failures are swallowed. We'll improve logging.

### Fix 3 — Remove `uvicorn[standard]` from requirements.txt

Vercel's runtime doesn't need uvicorn (it uses its own WSGI/ASGI adapter).
This reduces the bundle.

### Fix 4 — Add `query_points` filter kwarg to `search_by_text()`

`retriever.py` line 83: `search_by_text` builds `q_filter` but never passes
it to `query_points`. Minor bug but will cause wrong results.

---

## Verification Plan

1. Check which `__init__.py` files are present/missing
2. Apply fixes
3. Commit and redeploy
4. Hit `/health` endpoint — should return `{"status": "ok"}`

"""
api/__init__.py
===============
Public surface of the `api` package.

Importing `app` from here is the recommended way to run MemHub:

    uvicorn api:app --reload
"""

from api.main import app  # noqa: F401 — re-exported for `uvicorn api:app`

__all__ = ["app"]

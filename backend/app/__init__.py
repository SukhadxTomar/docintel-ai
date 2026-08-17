"""DocIntel-AI backend package.

Load ``backend/.env`` into the process environment as early as possible — before
any submodule is imported. ``pydantic-settings`` reads the same file to build the
typed ``settings`` object, but plain ``os.getenv`` consumers (e.g. the router
tunables in ``chains/router.py`` and ``LOG_FORMAT`` in ``utils/logger.py``) read
``os.environ`` directly, so the file's values must be present there too.

``load_dotenv`` does not override variables already set in the real environment,
so explicit env vars still win over ``.env``.
"""
from pathlib import Path

from dotenv import load_dotenv

# This file is backend/app/__init__.py, so parents[1] is the backend/ root.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

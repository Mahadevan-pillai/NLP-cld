"""MongoDB Atlas connection management.

Security rules followed here:
- The connection string is read ONLY from the environment variable
  named in config.yaml (default: MONGODB_URI). In local development
  python-dotenv loads it from the git-ignored .env file.
- The URI is never printed, logged, or put into exception messages.
  PyMongo's own error text can include host names or URI fragments, so
  errors are translated into short, generic, safe messages and raised
  with "from None" (the original exception is not chained).

Usage:
    from src.database.connection import get_connection
    conn = get_connection()      # one shared MongoClient per process
    conn.ping()
    db = conn.db                 # pymongo Database object
"""

import os
from dataclasses import dataclass, field
from datetime import timezone

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import (
    ConfigurationError,
    OperationFailure,
    PyMongoError,
    ServerSelectionTimeoutError,
)

from src.utils.config import REPO_ROOT, load_config

APP_NAME = "hinglish-meme-detection"


class DatabaseConfigError(Exception):
    """MongoDB configuration is missing or invalid (message is safe to show)."""


class DatabaseConnectionError(Exception):
    """Connecting to / talking with MongoDB failed (message is safe to show)."""


@dataclass(frozen=True)
class MongoSettings:
    # repr=False keeps the URI out of print(settings), logs and tracebacks.
    uri: str = field(repr=False)
    database_name: str
    server_selection_timeout_ms: int = 5000
    connect_timeout_ms: int = 10000
    socket_timeout_ms: int = 20000


def load_settings(config: dict | None = None) -> MongoSettings:
    """Build MongoSettings from config.yaml + environment variables."""
    # Fill os.environ from .env if present. override=False: real
    # environment variables (e.g. deployment secrets) take precedence.
    load_dotenv(REPO_ROOT / ".env", override=False)

    db_cfg = (config or load_config())["database"]
    uri_var = db_cfg["uri_env_var"]
    name_var = db_cfg["database_name_env_var"]

    uri = os.environ.get(uri_var, "").strip()
    if not uri:
        raise DatabaseConfigError(
            f"{uri_var} is not set. Add it to your local .env file "
            "(see .env.example and docs/database.md)."
        )
    if not uri.startswith(("mongodb://", "mongodb+srv://")):
        raise DatabaseConfigError(
            f"{uri_var} must start with 'mongodb://' or 'mongodb+srv://'."
        )

    database_name = (
        os.environ.get(name_var, "").strip() or db_cfg["default_database_name"]
    )

    return MongoSettings(
        uri=uri,
        database_name=database_name,
        server_selection_timeout_ms=db_cfg["server_selection_timeout_ms"],
        connect_timeout_ms=db_cfg["connect_timeout_ms"],
        socket_timeout_ms=db_cfg["socket_timeout_ms"],
    )


def safe_error_message(exc: Exception) -> str:
    """Translate a PyMongo exception into a generic, credential-free message."""
    if isinstance(exc, ServerSelectionTimeoutError):
        return (
            "could not reach the MongoDB server before the timeout. Check your "
            "internet connection, the Atlas Network Access list (your IP), and "
            "that the cluster is running."
        )
    if isinstance(exc, OperationFailure):
        # Standard MongoDB reports bad credentials as code 18; Atlas uses
        # its generic code 8000 ("AtlasError") with a "bad auth" message.
        errmsg = str((exc.details or {}).get("errmsg", "")).lower()
        if exc.code == 18 or (exc.code == 8000 and "bad auth" in errmsg):
            return "authentication failed. Check the database user's credentials in your .env."
        if getattr(exc, "code", None) == 13:
            return "the database user is not authorized for this operation."
        return f"database operation failed (code {getattr(exc, 'code', 'unknown')})."
    if isinstance(exc, ConfigurationError):
        return "the connection string is invalid or its DNS lookup failed."
    if isinstance(exc, PyMongoError):
        return f"MongoDB error ({type(exc).__name__})."
    return f"unexpected error ({type(exc).__name__})."


class MongoConnection:
    """Owns a single MongoClient and exposes the configured database.

    MongoClient keeps its own connection pool and is thread-safe, so one
    instance should be created per process and reused (not one per query).
    """

    def __init__(self, settings: MongoSettings | None = None):
        self.settings = settings or load_settings()
        self._client: MongoClient | None = None

    @property
    def client(self) -> MongoClient:
        if self._client is None:
            try:
                self._client = MongoClient(
                    self.settings.uri,
                    appname=APP_NAME,
                    serverSelectionTimeoutMS=self.settings.server_selection_timeout_ms,
                    connectTimeoutMS=self.settings.connect_timeout_ms,
                    socketTimeoutMS=self.settings.socket_timeout_ms,
                    # Return timezone-aware UTC datetimes instead of naive ones.
                    tz_aware=True,
                    tzinfo=timezone.utc,
                )
            except Exception as exc:  # e.g. DNS failure for mongodb+srv URIs
                raise DatabaseConnectionError(safe_error_message(exc)) from None
        return self._client

    @property
    def db(self):
        return self.client[self.settings.database_name]

    def ping(self) -> bool:
        """Round-trip to the server. Returns True or raises DatabaseConnectionError."""
        try:
            self.client.admin.command("ping")
        except DatabaseConnectionError:
            raise
        except Exception as exc:
            raise DatabaseConnectionError(safe_error_message(exc)) from None
        return True

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


_shared_connection: MongoConnection | None = None


def get_connection() -> MongoConnection:
    """Return the process-wide shared connection, creating it on first use."""
    global _shared_connection
    if _shared_connection is None:
        _shared_connection = MongoConnection()
    return _shared_connection


def close_connection() -> None:
    """Close the shared connection (e.g. at the end of a script)."""
    global _shared_connection
    if _shared_connection is not None:
        _shared_connection.close()
        _shared_connection = None

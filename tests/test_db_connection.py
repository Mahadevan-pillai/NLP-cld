"""Connection manager tests. MongoClient is mocked: no network, no Atlas.

The URI below is an obviously fake placeholder (".invalid" domain) used
only to prove that it never leaks into error messages or output.
"""

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pymongo.errors import ConfigurationError, OperationFailure, ServerSelectionTimeoutError

import src.database.connection as connection
from src.database.connection import (
    DatabaseConfigError,
    DatabaseConnectionError,
    MongoConnection,
    load_settings,
)

FAKE_PASSWORD = "fake-test-password-123"
FAKE_URI = f"mongodb+srv://fakeuser:{FAKE_PASSWORD}@cluster0.example.invalid/?retryWrites=true"
REPO_ROOT = Path(__file__).resolve().parents[1]


def assert_no_secret(text):
    assert FAKE_PASSWORD not in text
    assert FAKE_URI not in text
    assert "fakeuser" not in text
    assert "example.invalid" not in text


@pytest.fixture
def env_uri(monkeypatch):
    monkeypatch.setenv("MONGODB_URI", FAKE_URI)
    monkeypatch.delenv("MONGODB_DATABASE", raising=False)


@pytest.fixture
def mock_client(monkeypatch):
    client_cls = MagicMock()
    monkeypatch.setattr(connection, "MongoClient", client_cls)
    return client_cls


# ---------------- settings ----------------

def test_missing_uri_fails_clearly(monkeypatch):
    monkeypatch.delenv("MONGODB_URI", raising=False)
    with pytest.raises(DatabaseConfigError, match="MONGODB_URI is not set"):
        load_settings()


def test_blank_uri_treated_as_missing(monkeypatch):
    monkeypatch.setenv("MONGODB_URI", "   ")
    with pytest.raises(DatabaseConfigError):
        load_settings()


def test_invalid_scheme_error_does_not_echo_value(monkeypatch):
    monkeypatch.setenv("MONGODB_URI", f"http://fakeuser:{FAKE_PASSWORD}@example.invalid")
    with pytest.raises(DatabaseConfigError) as info:
        load_settings()
    assert_no_secret(str(info.value))


def test_default_database_name(env_uri):
    assert load_settings().database_name == "meme_detection_db"


def test_database_name_from_env(env_uri, monkeypatch):
    monkeypatch.setenv("MONGODB_DATABASE", "other_db")
    assert load_settings().database_name == "other_db"


def test_settings_repr_hides_uri(env_uri):
    settings = load_settings()
    assert_no_secret(repr(settings))
    assert_no_secret(str(settings))


# ---------------- MongoConnection ----------------

def test_client_created_once_with_safe_options(env_uri, mock_client):
    conn = MongoConnection()
    assert conn.client is conn.client
    mock_client.assert_called_once()
    kwargs = mock_client.call_args.kwargs
    assert kwargs["tz_aware"] is True
    assert kwargs["serverSelectionTimeoutMS"] == 5000


def test_db_uses_configured_name(env_uri, mock_client):
    conn = MongoConnection()
    conn.db
    mock_client.return_value.__getitem__.assert_called_with("meme_detection_db")


def test_ping_success(env_uri, mock_client):
    assert MongoConnection().ping() is True
    mock_client.return_value.admin.command.assert_called_with("ping")


@pytest.mark.parametrize(
    "exc, expected",
    [
        (ServerSelectionTimeoutError(f"timeout reaching {FAKE_URI}"), "could not reach"),
        (OperationFailure(f"bad auth for fakeuser:{FAKE_PASSWORD}", code=18), "authentication failed"),
        (OperationFailure("not allowed", code=13), "not authorized"),
    ],
)
def test_ping_failure_is_safe(env_uri, mock_client, exc, expected):
    mock_client.return_value.admin.command.side_effect = exc
    with pytest.raises(DatabaseConnectionError) as info:
        MongoConnection().ping()
    message = str(info.value)
    assert expected in message
    assert_no_secret(message)
    # Original exception (which may contain the URI) is not chained.
    assert info.value.__cause__ is None and info.value.__suppress_context__


def test_client_construction_failure_is_safe(env_uri, mock_client):
    mock_client.side_effect = ConfigurationError(f"DNS failed for {FAKE_URI}")
    with pytest.raises(DatabaseConnectionError) as info:
        MongoConnection().client
    assert_no_secret(str(info.value))


def test_close_and_context_manager(env_uri, mock_client):
    with MongoConnection() as conn:
        conn.ping()
    mock_client.return_value.close.assert_called_once()
    assert conn._client is None


def test_shared_connection_reused_and_closed(env_uri, mock_client):
    first = connection.get_connection()
    assert connection.get_connection() is first
    first.client
    connection.close_connection()
    mock_client.return_value.close.assert_called_once()
    assert connection.get_connection() is not first


# ---------------- scripts/check_mongodb.py ----------------

def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_script_missing_uri(monkeypatch, capsys):
    monkeypatch.delenv("MONGODB_URI", raising=False)
    assert load_script("check_mongodb").main() == 1
    assert "MongoDB connection failed: MONGODB_URI is not set" in capsys.readouterr().out


def test_check_script_success_output_is_safe(env_uri, mock_client, capsys):
    mock_client.return_value.__getitem__.return_value.list_collection_names.return_value = []
    assert load_script("check_mongodb").main() == 0
    out = capsys.readouterr().out
    assert "MongoDB connection successful." in out
    assert_no_secret(out)


def test_check_script_failure_output_is_safe(env_uri, mock_client, capsys):
    mock_client.return_value.admin.command.side_effect = OperationFailure(
        f"auth failed {FAKE_URI}", code=18
    )
    assert load_script("check_mongodb").main() == 1
    out = capsys.readouterr().out
    assert "MongoDB connection failed: " in out
    assert_no_secret(out)
    assert out.count("MongoDB connection failed") == 1  # prefix not duplicated


def test_init_script_runs_ensure_indexes(env_uri, mock_client, capsys):
    db = mock_client.return_value.__getitem__.return_value
    db.__getitem__.return_value.create_index.side_effect = lambda keys, **kw: kw["name"]
    assert load_script("init_database").main() == 0
    out = capsys.readouterr().out
    assert "initialized" in out and "analysis_id_unique" in out
    assert_no_secret(out)

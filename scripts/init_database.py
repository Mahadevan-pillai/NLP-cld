"""Initialize the MongoDB database: create the indexes (idempotent).

Run from the repository root:
    python scripts/init_database.py

No documents are inserted. MongoDB creates a collection lazily, the
first time something is written to it; creating an index counts as a
write, so after this script the three collections exist (empty).
Safe to run more than once.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database.connection import (  # noqa: E402
    DatabaseConfigError,
    DatabaseConnectionError,
    MongoConnection,
    safe_error_message,
)
from src.database.indexes import ensure_indexes  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main() -> int:
    try:
        config = load_config()
        with MongoConnection() as conn:
            conn.ping()
            created = ensure_indexes(conn.db, config["database"]["collections"])
            print(f"Database '{conn.settings.database_name}' initialized.")
            for collection, index_names in created.items():
                print(f"  {collection}: {', '.join(index_names)}")
        return 0
    except (DatabaseConfigError, DatabaseConnectionError) as exc:
        print(f"Database initialization failed: {exc}")
    except Exception as exc:
        print(f"Database initialization failed: {safe_error_message(exc)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

"""Check that MongoDB Atlas is reachable with the configured credentials.

Run from the repository root:
    python scripts/check_mongodb.py

Exit code 0 = connected, 1 = failed. Never prints the URI or credentials.
"""

import sys
from pathlib import Path

# Make "src" importable when this file is run directly as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.database.connection import (  # noqa: E402
    DatabaseConfigError,
    DatabaseConnectionError,
    MongoConnection,
    safe_error_message,
)


def main() -> int:
    try:
        with MongoConnection() as conn:
            conn.ping()
            # Listing collections proves the user can access this database
            # (ping alone only proves the server is reachable).
            collections = conn.db.list_collection_names()
            print("MongoDB connection successful.")
            print(f"Database: {conn.settings.database_name} "
                  f"({len(collections)} existing collection(s))")
        return 0
    except (DatabaseConfigError, DatabaseConnectionError) as exc:
        print(f"MongoDB connection failed: {exc}")
    except Exception as exc:  # e.g. not authorized to list collections
        print(f"MongoDB connection failed: {safe_error_message(exc)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

"""Index definitions for the application collections.

ensure_indexes() is idempotent: MongoDB's create_index does nothing if
an identical index already exists, so it is safe to run repeatedly.

Why these indexes:
- unique *_id fields: fast lookup by application ID and a guarantee
  against duplicates.
- created_at (descending): "most recent first" listings.
- (type, created_at) compound indexes: serve both "filter by type" and
  "filter by type, newest first". A compound index also answers queries
  on its first field alone, so a separate single-field type index
  would be redundant.
- (analysis_id, created_at) on feedback: "all feedback for an analysis,
  newest first"; also serves lookups by analysis_id alone.
"""

from pymongo import ASCENDING, DESCENDING

from src.database.repositories import DEFAULT_COLLECTIONS

INDEXES = {
    "analysis_history": [
        ([("analysis_id", ASCENDING)], {"name": "analysis_id_unique", "unique": True}),
        ([("created_at", DESCENDING)], {"name": "created_at_desc"}),
        ([("analysis_type", ASCENDING), ("created_at", DESCENDING)],
         {"name": "analysis_type_created_at"}),
    ],
    "user_feedback": [
        ([("feedback_id", ASCENDING)], {"name": "feedback_id_unique", "unique": True}),
        ([("analysis_id", ASCENDING), ("created_at", DESCENDING)],
         {"name": "analysis_id_created_at"}),
        ([("created_at", DESCENDING)], {"name": "created_at_desc"}),
    ],
    "model_versions": [
        ([("model_id", ASCENDING)], {"name": "model_id_unique", "unique": True}),
        ([("model_type", ASCENDING), ("created_at", DESCENDING)],
         {"name": "model_type_created_at"}),
        ([("created_at", DESCENDING)], {"name": "created_at_desc"}),
    ],
}


def ensure_indexes(db, collection_names: dict | None = None) -> dict[str, list[str]]:
    """Create all indexes (if missing). Returns {collection: [index names]}."""
    names = {**DEFAULT_COLLECTIONS, **(collection_names or {})}
    created = {}
    for logical_name, index_specs in INDEXES.items():
        collection = db[names[logical_name]]
        created[names[logical_name]] = [
            collection.create_index(keys, **options) for keys, options in index_specs
        ]
    return created

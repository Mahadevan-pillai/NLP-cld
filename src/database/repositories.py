"""Repository classes: the only place that runs MongoDB queries.

Future Streamlit / ML code should call these methods instead of using
PyMongo directly, so database details stay in one place.

Each repository receives a pymongo Database (e.g. get_connection().db),
so one shared MongoClient is reused by all of them.
Returned documents exclude MongoDB's internal "_id"; the application
uses its own UUID fields (analysis_id, feedback_id, model_id).
"""

from pymongo import DESCENDING

from src.database import schemas

DEFAULT_COLLECTIONS = {
    "analysis_history": "analysis_history",
    "user_feedback": "user_feedback",
    "model_versions": "model_versions",
}
HIDE_ID = {"_id": 0}
MAX_LIST_LIMIT = 100


def _check_limit(limit: int) -> int:
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_LIST_LIMIT:
        raise ValueError(f"limit must be an integer between 1 and {MAX_LIST_LIMIT}.")
    return limit


def _without_id(doc: dict) -> dict:
    return {k: v for k, v in doc.items() if k != "_id"}


class AnalysisHistoryRepository:
    def __init__(self, db, collection_name=None, allowed_labels=None):
        self.collection = db[collection_name or DEFAULT_COLLECTIONS["analysis_history"]]
        self.allowed_labels = allowed_labels

    def create_analysis(
        self,
        analysis_type: str,
        input_data: dict,
        predictions: dict | None = None,
        model_versions: dict | None = None,
        processing_time_ms: float | None = None,
    ) -> dict:
        """Validate and store a new analysis. Returns the stored document."""
        doc = schemas.build_analysis(
            analysis_type, input_data, predictions, model_versions,
            processing_time_ms, allowed_labels=self.allowed_labels,
        )
        # insert_one adds "_id" to the dict it is given, so pass a copy.
        self.collection.insert_one(dict(doc))
        return doc

    def get_analysis_by_id(self, analysis_id: str) -> dict | None:
        return self.collection.find_one({"analysis_id": analysis_id}, HIDE_ID)

    def list_recent_analyses(self, limit: int = 20, analysis_type: str | None = None) -> list[dict]:
        query = {}
        if analysis_type is not None:
            if analysis_type not in schemas.ANALYSIS_TYPES:
                raise ValueError(f"analysis_type must be one of {schemas.ANALYSIS_TYPES}.")
            query["analysis_type"] = analysis_type
        cursor = self.collection.find(query, HIDE_ID).sort("created_at", DESCENDING)
        return list(cursor.limit(_check_limit(limit)))

    def exists(self, analysis_id: str) -> bool:
        return self.collection.count_documents({"analysis_id": analysis_id}, limit=1) > 0


class UserFeedbackRepository:
    def __init__(self, db, collection_name=None, analysis_repository=None):
        self.collection = db[collection_name or DEFAULT_COLLECTIONS["user_feedback"]]
        # Used to refuse feedback for analyses that do not exist.
        self.analyses = analysis_repository or AnalysisHistoryRepository(db)

    def create_feedback(self, analysis_id: str, rating: str, comment: str | None = None) -> dict:
        doc = schemas.build_feedback(analysis_id, rating, comment)
        if not self.analyses.exists(analysis_id):
            raise ValueError("Cannot store feedback: analysis_id does not exist.")
        self.collection.insert_one(dict(doc))
        return doc

    def get_feedback_for_analysis(self, analysis_id: str) -> list[dict]:
        cursor = self.collection.find({"analysis_id": analysis_id}, HIDE_ID)
        return list(cursor.sort("created_at", DESCENDING))


class ModelVersionRepository:
    def __init__(self, db, collection_name=None):
        self.collection = db[collection_name or DEFAULT_COLLECTIONS["model_versions"]]

    def register_model_version(
        self,
        model_type: str,
        model_name: str,
        version: str,
        architecture: str | None = None,
        training_dataset: str | None = None,
        preprocessing_version: str | None = None,
        metrics_metadata: dict | None = None,
    ) -> dict:
        """Record a trained model. Call only for models that actually exist."""
        doc = schemas.build_model_version(
            model_type, model_name, version, architecture,
            training_dataset, preprocessing_version, metrics_metadata,
        )
        self.collection.insert_one(dict(doc))
        return doc

    def get_model_version(self, model_id: str) -> dict | None:
        return self.collection.find_one({"model_id": model_id}, HIDE_ID)

    def list_model_versions(self, model_type: str | None = None, limit: int = 50) -> list[dict]:
        query = {}
        if model_type is not None:
            if model_type not in schemas.MODEL_TYPES:
                raise ValueError(f"model_type must be one of {schemas.MODEL_TYPES}.")
            query["model_type"] = model_type
        cursor = self.collection.find(query, HIDE_ID).sort("created_at", DESCENDING)
        return list(cursor.limit(_check_limit(limit)))

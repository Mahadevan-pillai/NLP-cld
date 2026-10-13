"""Shared test fixtures.

Database tests must never touch the real .env or MongoDB Atlas, so:
- load_dotenv is disabled for every test (autouse fixture);
- FakeDatabase is a tiny in-memory stand-in for a pymongo Database,
  supporting only the operations the repositories use.
"""

import copy
import itertools

import pytest


@pytest.fixture(autouse=True)
def no_dotenv(monkeypatch):
    """Stop tests from reading a real MONGODB_URI out of the local .env."""
    import src.database.connection as connection

    monkeypatch.setattr(connection, "load_dotenv", lambda *args, **kwargs: False)
    monkeypatch.setattr(connection, "_shared_connection", None)


class FakeCursor:
    def __init__(self, docs):
        self._docs = docs

    def sort(self, key, direction):
        self._docs = sorted(self._docs, key=lambda d: d[key], reverse=direction == -1)
        return self

    def limit(self, n):
        self._docs = self._docs[:n]
        return self

    def __iter__(self):
        return iter(self._docs)


class FakeCollection:
    _ids = itertools.count(1)

    def __init__(self):
        self.docs = []
        self.indexes = {}

    @staticmethod
    def _matches(doc, query):
        return all(doc.get(k) == v for k, v in query.items())

    @staticmethod
    def _project(doc, projection):
        doc = copy.deepcopy(doc)
        if projection == {"_id": 0}:
            doc.pop("_id", None)
        return doc

    def insert_one(self, doc):
        doc.setdefault("_id", next(self._ids))  # mimic pymongo mutating the dict
        self.docs.append(copy.deepcopy(doc))

    def find_one(self, query, projection=None):
        for doc in self.docs:
            if self._matches(doc, query):
                return self._project(doc, projection)
        return None

    def find(self, query, projection=None):
        return FakeCursor([self._project(d, projection) for d in self.docs if self._matches(d, query)])

    def count_documents(self, query, limit=0):
        count = sum(1 for d in self.docs if self._matches(d, query))
        return min(count, limit) if limit else count

    def create_index(self, keys, name, unique=False):
        self.indexes[name] = {"keys": keys, "unique": unique}
        return name


class FakeDatabase:
    def __init__(self):
        self.collections = {}

    def __getitem__(self, name):
        return self.collections.setdefault(name, FakeCollection())

    def list_collection_names(self):
        return list(self.collections)


@pytest.fixture
def fake_db():
    return FakeDatabase()

"""Small file-output helpers shared by the scripts."""

import json
from pathlib import Path

VOLATILE_KEYS = {"generated_at"}


def write_json_if_changed(path: Path, data: dict) -> str:
    """Write JSON unless only volatile keys (timestamps) would change.

    Keeps re-runs from showing spurious Git changes. Returns
    "written" or "unchanged".
    """
    path = Path(path)
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        strip = lambda d: {k: v for k, v in d.items() if k not in VOLATILE_KEYS}  # noqa: E731
        if strip(old) == strip(data):
            return "unchanged"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return "written"

from typing import Any
import hashlib
import json


def sha256_hash(obj: Any):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()

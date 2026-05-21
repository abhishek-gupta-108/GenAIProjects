from pathlib import Path
from typing import Any

import chromadb
import config
from chromadb.api.models.Collection import Collection

COLLECTION_NAME = "resume_embeddings"
_collection = None


def init_collection() -> Collection:
    Path(config.CHROMA_PATH).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    _collection = client.get_or_create_collection(name=COLLECTION_NAME)
    return _collection


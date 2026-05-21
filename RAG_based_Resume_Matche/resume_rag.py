import sys
import json
import chromadb
import config
from datasets import load_dataset
from pathlib import Path

from extractor import parse_resume, _init_resume_db
from vector_store import init_collection
from config import build_Azure_embedding_client
from llama_index.core import Document, StorageContext
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
#from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline, IngestionCache
from llama_index.vector_stores.chroma import ChromaVectorStore
from chromadb.api.models.Collection import Collection




def ingest(raw_resume_list: list) -> None:
    """
        Raw Resumes
                ↓
        [Loading]           ← from HuggingFace dataset => Document
        [Transformations]   ← chunk, embed => nodes
                ↓
        [Ingestion]        ← run and cache the pipeline
                ↓
        [VectorStore]      ← ChromaDB (persistent storage of nodes)
    """

    # Step 1: LOADING - Create a Document for LlamaIndex
    # https://developers.llamaindex.ai/python/framework/module_guides/loading/documents_and_nodes/

    llama_documents = [
       parse_resume(raw_text, source_ref) for raw_text, source_ref in raw_resume_list
    ]
   
    # Step 2 - Ingestion Pipeline: chunking, embedding, and caching, storing the final nodes in ChromaDB

    # assign chroma as the vector_store to the context
    chroma_collection = init_collection()
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    pipeline = IngestionPipeline(
        transformations=[
            #SentenceSplitter(chunk_size=1, chunk_overlap=0), #Intentionally set chunk_size=1 to treat the whole resume as one chunk for embedding.
            # TitleExtractor(),
            build_Azure_embedding_client(),
        ],
             vector_store=vector_store   # ← pipeline writes directly to ChromaDB
    )
    # pipeline.persist("./data/pipeline_storage")
    nodes = pipeline.run(documents=llama_documents)

    # Debug Collection
    print(chroma_collection.count())
    result = chroma_collection.peek(limit=config.MAX_RESUMES)
    for i in range(len(result["ids"])):
        print(f"ID: {result['ids'][i]}")
        print(f"Text: {result['documents'][i][:100]}...")  # first 100 chars
        print(f"Embedding: {result['embeddings'][i]}")
        print(f"Metadata: {result['metadatas'][i]}")
        print("---")
    

def main() -> None:
    # Initialize the resume database
    _init_resume_db()
    
    print(
        f"Loading dataset {config.HF_RESUME_DATASET} (split={config.HF_RESUME_SPLIT}, text_field={config.HF_RESUME_TEXT_FIELD})"
    )
    dataset = load_dataset(config.HF_RESUME_DATASET, split=config.HF_RESUME_SPLIT)

    ingested = 0
    raw_resume_list = []
    for idx, row in enumerate(dataset):
        resume_text = str(row.get(config.HF_RESUME_TEXT_FIELD, "")).strip()
        if not resume_text:
            continue

        source_ref = f"hf://{config.HF_RESUME_DATASET}/{config.HF_RESUME_SPLIT}/{idx}"

        raw_resume_list.append((resume_text, source_ref))
        
        ingested += 1

        if ingested >= config.MAX_RESUMES:
            break

    ingest(raw_resume_list)
    print(f"Completed ingestion for {ingested} resumes.")


if __name__ == "__main__":
    main()

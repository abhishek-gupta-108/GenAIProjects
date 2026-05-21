import json
import importlib
import re
import sys
from typing import Any

from llama_index.core import Settings, VectorStoreIndex

import config
from config import build_Azure_embedding_client, build_Azure_OpenAI_client
from datasets import load_dataset
from llama_index.vector_stores.chroma import ChromaVectorStore
from vector_store import init_collection
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.llms.azure_openai import AzureOpenAI as LlamaAzureOpenAI
from extractor import get_resume_from_db, _init_resume_db


# Update LLamaIndex's global Setting config
Settings.embed_model = build_Azure_embedding_client()
Settings.llm = llm = build_Azure_OpenAI_client()

def match_candidate_for_job(jd_text: str, must_have_skills: str) -> dict[str, Any]:
    chroma_collection = init_collection()
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(vector_store)

    # semantically retrieve top-k candidates for the given job description
    top_k = config.JOB_MATCH_TOP_K
    vector_retriever = index.as_retriever(similarity_top_k=top_k * 2)  # Retrieve more to filter

    # First retrieve a larger pool of candidates using semantic search
    nodes_with_semantic_search = vector_retriever.retrieve(
        "Based on the following job description, find the most relevant candidates from the resume database. " + jd_text
    )

    # If the must have skills are specified, filter the candidates using BM25 keyword search
    if must_have_skills:
        # BM25 retreiver to score based on exact keyword matches.
        bm25_on_pool = BM25Retriever.from_defaults(
            nodes=[n.node for n in nodes_with_semantic_search],
            similarity_top_k=top_k,
            tokenizer=lambda text: text.lower().split(separators=[","])
        )
        nodes_with_best_match_search = bm25_on_pool.retrieve(must_have_skills)
        result_nodes = nodes_with_best_match_search

        if len(result_nodes) < top_k:
            ## sort the nodes_with_semantic_search based on its score and fill in the rest of the top_k results if BM25 filtering results in less than top_k candidates
            remaining_slots = top_k - len(result_nodes)
            sorted_semantic_nodes = sorted(nodes_with_semantic_search, key=lambda x: x.score or 0.0, reverse=True)
            for node in sorted_semantic_nodes:
                if node not in result_nodes:
                    result_nodes.append(node)
                if len(result_nodes) >= top_k:
                    break
    else:
        result_nodes = nodes_with_semantic_search
    
    
    ## Intentionally not doing fusion search because that will send in the entire JD to both retrievers
    #fusion_retriever = QueryFusionRetriever(
        #    retrievers=[vector_retriever, bm25_retriever],
        #    mode="reciprocal_rerank",
        #    similarity_top_k=top_k,
        #    num_queries=3,
        #)
    
    top_matches: list[dict[str, Any]] = []

    for item in result_nodes:
        node = item.node
        score = float(item.score or 0.0)

        node_metadata = getattr(node, "metadata", {}) or {}
        node_metadata_userid = node_metadata.get("user_id", "N/A")
        
        # Fetch resume from database
        resume = get_resume_from_db(node_metadata_userid)

        resume_payload: dict[str, Any] = {}
        if resume is not None:
            resume_payload = resume.model_dump()

        match: dict[str, Any] = {
            "score": score * 100,  # convert to percentage
            "resume": resume_payload,
        }
        top_matches.append(match)

        if len(top_matches) >= top_k:
            break

    return {
        "job_description": jd_text,
        "top_matches": top_matches,
    }




def main() -> None:
    # Initialize the database
    _init_resume_db()

    # Ask user for exact skill matching
    bool_provide_skills = input("Do you want to provide must-have skills for filtering? (y/n): ").strip().lower()
    skills_input = ""
    if bool_provide_skills == 'y':
        skills_input = input("Enter the must-have skills, separated by commas: ")
    
    dataset = load_dataset(config.HF_JD_DATASET, split=config.HF_JD_SPLIT)
    ingested = 0

    for idx, row in enumerate(dataset):
        jd_text = "HIRING COMPANY: " + str(row.get(config.HF_JD_COMPANY_FIELD, "")).strip() + "\n"
        jd_text += "HIRING POSITION: " + str(row.get(config.HF_JD_POSITION_FIELD, "")).strip() + "\n"
        jd_text += "JOB DESCRIPTION: " + str(row.get(config.HF_JD_JD_FIELD, "")).strip()
        if not jd_text:
            continue

        match_result = match_candidate_for_job(jd_text, skills_input)
        print("\n" + "=" * 100)
        print(f"JOB #{ingested + 1} MATCH RESULTS")
        print("=" * 100)
        print(json.dumps(match_result, indent=2, ensure_ascii=False))

        ingested += 1

        if ingested >= config.MAX_JDS:
            break

    print(f"Completed ingestion for {ingested} job descriptions")

if __name__ == "__main__":
    main()

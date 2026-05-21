import os

from pydantic import config
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI as LlamaAzureOpenAI
#from dotenv import load_dotenv

#load_dotenv()

AZURE_ENDPOINT: str = os.environ["AZURE_OPENAI_ENDPOINT_CA"]
AZURE_API_KEY: str = os.environ["AZURE_OPENAI_KEY_CA"]

## Chat model name
CHAT_DEPLOYMENT: str = "gpt-4o"
CHAT_API_VERSION: str = "2024-12-01-preview"


## Embedding model name (must be an Azure OpenAI deployment)
EMBEDDING_DEPLOYMENT: str = "text-embedding-3-small"
EMBEDDINNG_API_VERSION: str = "2024-02-01"

CHROMA_PATH: str = os.getenv("CHROMA_PATH", "data/chroma")
JOB_MATCH_TOP_K: int = int(os.getenv("JOB_MATCH_TOP_K", "10"))

HF_RESUME_DATASET: str = "AzharAli05/Resume-Screening-Dataset"
HF_RESUME_SPLIT: str = "train"
HF_RESUME_TEXT_FIELD: str = "Resume"
MAX_RESUMES:int = 30

# https://huggingface.co/datasets/jacob-hugging-face/job-descriptions
HF_JD_DATASET: str = "jacob-hugging-face/job-descriptions"
HF_JD_SPLIT: str = "train"
HF_JD_COMPANY_FIELD: str = "company_name"
HF_JD_JD_FIELD: str = "job_description"
HF_JD_POSITION_FIELD: str = "position_title"
MAX_JDS: int = 1

def build_Azure_embedding_client() -> AzureOpenAIEmbedding:
    return  AzureOpenAIEmbedding(
                azure_endpoint=AZURE_ENDPOINT,
                api_key=AZURE_API_KEY,
                api_version=EMBEDDINNG_API_VERSION,
                model=EMBEDDING_DEPLOYMENT,
            )

def build_Azure_OpenAI_client() -> LlamaAzureOpenAI:
    return LlamaAzureOpenAI(
        model=CHAT_DEPLOYMENT,
        engine=CHAT_DEPLOYMENT,
        api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_ENDPOINT,
        api_version=CHAT_API_VERSION,
        temperature=0,
    )

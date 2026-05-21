# RAG-based Resume Matcher

This project ingests resume text, extracts structured resume data with Azure OpenAI, stores the structured payload in SQLite, and stores embeddings in ChromaDB for semantic matching against job descriptions.

The matching flow supports:

- semantic retrieval with LlamaIndex + ChromaDB
- optional exact must-have skill filtering from the user
- pretty-printed job match output in the terminal
- full ResumeSchema lookup from SQLite during job matching

## How It Works

1. `resume_rag.py` loads resume text from the Hugging Face dataset.
2. `extractor.py` converts each resume into a `ResumeSchema` object.
3. The parsed resume is stored in SQLite as JSON under its `user_id`.
4. The same resume is embedded and written to ChromaDB.
5. `job_matcher.py` loads job descriptions, retrieves top matches, and prints the results.
6. If enabled, the matcher asks for comma-separated must-have skills and filters candidates by exact skill match.
7. Each ChromaDB record stores the `user_id` in metadata so job matching can use it to fetch the full resume JSON from SQLite later.

## Resources referred to
LLamaHub - to get the LLMs, Extractirs, Indexes, Storagem Post Processors, Embeddings info - https://llamahub.ai/

LLamnaIndex - Getting Started for RAG pipeline - https://developers.llamaindex.ai/python/framework/understanding/rag/
LLamaIndex Documentation - for Indexing, Loading, Querying, Retrievers, Node processors, Structured Output - https://developers.llamaindex.ai/python/framework/module_guides/querying/retriever/

HuggingFace DataSets-
Resumes - https://huggingface.co/datasets/AzharAli05/Resume-Screening-Dataset
Job Description - https://huggingface.co/datasets/jacob-hugging-face/job-descriptions



## Prerequisites

- Python 3.10+
- An Azure OpenAI resource with:
  - a chat deployment
  - an embedding deployment

## Setup

1. Install dependencies.

   ```bash
   pip install -r requirements.txt
   ```

2. Configure environment variables.

   The project expects these variables:

   - `AZURE_OPENAI_ENDPOINT_CA`
   - `AZURE_OPENAI_KEY_CA`

   Optional variables:

   - `CHROMA_PATH` (defaults to `data/chroma`)
   - `JOB_MATCH_TOP_K` (defaults to `10`)

3. Make sure your Azure OpenAI deployments in `config.py` match your resource names.

## Repository Layout

| File | Purpose |
|---|---|
| `config.py` | Environment and model configuration |
| `extractor.py` | Resume schema, parsing, and SQLite storage |
| `resume_rag.py` | Resume ingestion pipeline |
| `job_matcher.py` | Job description matching and terminal output |
| `vector_store.py` | ChromaDB collection setup and embeddings |

## Ingest Resumes

Run the resume ingestion script to parse and store resumes:

```bash
python resume_rag.py
```

What this does:

- loads resumes from `config.HF_RESUME_DATASET`
- extracts structured fields into `ResumeSchema`
- stores `user_id` + JSON dump in SQLite at `data/resumes.db`
- stores embeddings in ChromaDB at `data/chroma/`

## Match Jobs

Run the matcher script:

```bash
python job_matcher.py
```

At startup, the script asks:

1. Do you want to provide must-have skills for filtering? (`y/n`)
2. If yes, enter a comma-separated list of must-have skills

Then it:

- loads job descriptions from `config.HF_JD_DATASET`
- retrieves the top semantic matches from ChromaDB
- optionally filters candidates by exact must-have skills
- fetches the full resume JSON from SQLite by `user_id`
- pretty prints the job description and all matched candidates in the terminal

## Data Storage

The project stores data in two persistent locations:

- `data/resumes.db` - SQLite database containing `user_id` and JSON dump of each `ResumeSchema`
- `data/chroma/` - ChromaDB vector store containing resume embeddings and `user_id` metadata used to look up the resume in SQLite during job matching

## Resume Schema

The structured resume model is defined in `extractor.py` and includes:

- `name`
- `summary`
- `total_years_of_experience`
- `skills`
- `interests`
- `contact`
- `education`
- `job_experience`

## Matching Notes

- Exact skill matching is case-insensitive.
- If must-have skills are provided, a candidate is kept only when all required skills are present.
- Candidate details are loaded from SQLite, so job matching works across runs.

## Troubleshooting

- If matching returns empty results, confirm that `resume_rag.py` has been run first.
- If SQLite does not contain records, delete `data/resumes.db` and re-run ingestion.
- If Azure OpenAI calls fail, check the endpoint, key, and deployment names in your environment and `config.py`.

## Example Workflow

```bash
python resume_rag.py
python job_matcher.py
```

First ingest resumes, then run the matcher to print job-to-resume results.

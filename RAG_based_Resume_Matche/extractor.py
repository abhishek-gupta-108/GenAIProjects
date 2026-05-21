import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from pydantic import BaseModel, Field

from config import AZURE_API_KEY, AZURE_ENDPOINT, CHAT_API_VERSION, CHAT_DEPLOYMENT
from config import build_Azure_OpenAI_client

from llama_index.core import Document

# SQLite database for storing resumes
RESUME_DB_PATH = "data/resumes.db"

class ContactInfo(BaseModel):
    phone: str = ""
    linkedin: str = ""
    email: str = ""


class EducationEntry(BaseModel):
    institute_name: str = ""
    course: str = ""
    start_year: str = ""
    end_year: str = ""


class JobExperienceEntry(BaseModel):
    company_name: str = ""
    position: str = ""
    job_designation: str = ""
    job_industry: str = ""
    start_date: str = ""
    end_date: str = ""
    skills_used: list[str] = Field(default_factory=list)
    details: list[str] = Field(default_factory=list)


class ResumeSchema(BaseModel):
    name: str = ""
    summary: str = ""
    total_years_of_experience: float = 0.0
    skills: list[str] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)
    contact: ContactInfo = Field(default_factory=ContactInfo)
    education: list[EducationEntry] = Field(default_factory=list)
    job_experience: list[JobExperienceEntry] = Field(default_factory=list)



def _init_resume_db() -> None:
    """Initialize SQLite database for storing resumes."""
    Path(RESUME_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(RESUME_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS resumes (
            user_id TEXT PRIMARY KEY,
            resume_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def _store_resume_in_db(user_id: str, resume: ResumeSchema) -> None:
    """Store resume data in SQLite database."""
    try:
        conn = sqlite3.connect(RESUME_DB_PATH)
        cursor = conn.cursor()
        resume_json = json.dumps(resume.model_dump())
        cursor.execute(
            "INSERT OR REPLACE INTO resumes (user_id, resume_json) VALUES (?, ?)",
            (user_id, resume_json),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[WARN] Failed to store resume in DB for user_id {user_id}: {e}")


def get_resume_from_db(user_id: str) -> ResumeSchema | None:
    """Retrieve resume data from SQLite database by user_id."""
    try:
        conn = sqlite3.connect(RESUME_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT resume_json FROM resumes WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            resume_data = json.loads(row[0])
            return ResumeSchema(**resume_data)
    except Exception as e:
        print(f"[WARN] Failed to retrieve resume from DB for user_id {user_id}: {e}")
    return None


def _generate_user_id(name: str, phone: str) -> str:
    value = f"{name.strip().lower()}|{phone.strip()}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def create_llama_document(resume: ResumeSchema, user_id: str, source_ref: str) -> Document:
    return Document(
        text=json.dumps(resume.model_dump(), indent=2),
        metadata={"user_id": user_id, "source_ref": source_ref},
    )


def parse_resume(resume_content: str, source_ref: str) -> Document:
    # https://llamahub.ai/l/llms/llama-index-llms-azure-openai?from=llms
    llm = build_Azure_OpenAI_client()

    # https://developers.llamaindex.ai/python/framework/understanding/extraction/structured_llms/
    sllm = llm.as_structured_llm(ResumeSchema)

    response = sllm.complete(resume_content)
    final_resume = response.raw

    user_id = _generate_user_id(final_resume.name, final_resume.contact.phone)
    
    # Store resume in SQLite database
    _store_resume_in_db(user_id, final_resume)
    
    document = create_llama_document(final_resume, user_id, source_ref)
    return document

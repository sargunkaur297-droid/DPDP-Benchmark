
"""Configuration for the DPDP RAG pipeline."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXT_DIR = ROOT / "data" / "text"
OFFICIAL_TEXT_DIR = TEXT_DIR / "official"
DB_DIR = ROOT / "chroma_db"
COLLECTION = "dpdp"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHAT_MODEL = "llama-3.3-70b-versatile"

MAX_OUTPUT_TOKENS = int(os.environ.get("DPDP_MAX_OUTPUT_TOKENS", "2000"))

CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 8
EMBED_BATCH = 8

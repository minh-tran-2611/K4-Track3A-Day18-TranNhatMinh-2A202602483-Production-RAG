"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# --- LLM provider (OpenAI-compatible) ---
# Ưu tiên Gemini (qua endpoint OpenAI-compatible) nếu có GEMINI_API_KEY, ngược lại dùng OpenAI.
if GEMINI_API_KEY:
    LLM_API_KEY = GEMINI_API_KEY
    LLM_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    LLM_MODEL = "gemini-3.8-flash"
    LLM_EXTRA_PARAMS = {"reasoning_effort": "none"}  # tắt thinking: nhanh hơn, không ăn max_tokens
    EVAL_EMBEDDING_MODEL = "gemini-embedding-001"
else:
    LLM_API_KEY = OPENAI_API_KEY
    LLM_BASE_URL = None
    LLM_MODEL = "gpt-4o-mini"
    LLM_EXTRA_PARAMS = {}
    EVAL_EMBEDDING_MODEL = "text-embedding-3-small"


def get_llm_client():
    """OpenAI SDK client trỏ tới provider đang cấu hình (OpenAI hoặc Gemini)."""
    from openai import OpenAI
    return OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, max_retries=6)

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")

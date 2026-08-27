"""Centralized config so every module reads settings the same way."""
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.4"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_db")

INJECTION_RISK_THRESHOLD = float(os.getenv("INJECTION_RISK_THRESHOLD", "0.6"))

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))
ROUTER_TOP_K = int(os.getenv("ROUTER_TOP_K", "2"))

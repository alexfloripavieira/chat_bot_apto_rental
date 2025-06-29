import os
from dotenv import load_dotenv

load_dotenv()

# Configurações do Modelo de Linguagem (LLM)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
OPENAI_MODEL_TEMPERATURE = float(os.getenv("OPENAI_MODEL_TEMPERATURE", 0.0))

# Prompts do Sistema
AI_CONTEXTUALIZE_PROMPT = os.getenv("AI_CONTEXTUALIZE_PROMPT")
AI_SYSTEM_PROMPT = os.getenv("AI_SYSTEM_PROMPT")

# Configurações da Evolution API
EVOLUTION_API_URL = os.getenv("EVOLUTION_API_URL")
EVOLUTION_INSTANCE_NAME = os.getenv("EVOLUTION_INSTANCE_NAME")
EVOLUTION_AUTHENTICATION_API_KEY = os.getenv("AUTHENTICATION_API_KEY")

# Configurações do Banco de Dados e Cache (se usado pela Evolution)
DATABASE_ENABLED = os.getenv("DATABASE_ENABLED", "true").lower() == "true"
DATABASE_PROVIDER = os.getenv("DATABASE_PROVIDER")
DATABASE_CONNECTION_URI = os.getenv("DATABASE_CONNECTION_URI")
CACHE_REDIS_ENABLED = os.getenv("CACHE_REDIS_ENABLED", "true").lower() == "true"
CACHE_REDIS_URI = os.getenv("CACHE_REDIS_URI")

# Configurações do RAG (Retrieval-Augmented Generation)
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "vectorstore")
RAG_FILES_DIR = os.getenv("RAG_FILES_DIR", "rag_files")

# URL para o scraping de imóveis
WEB_PAGE_URL = os.getenv("WEB_PAGE_URL")

# Credenciais de Admin (se necessário para o scraping)
REAL_ESTATE_ADMIN_URL = os.getenv("REAL_ESTATE_ADMIN_URL")
REAL_ESTATE_ADMIN_USERNAME = os.getenv("REAL_ESTATE_ADMIN_USERNAME")
REAL_ESTATE_ADMIN_PASSWORD = os.getenv("REAL_ESTATE_ADMIN_PASSWORD")
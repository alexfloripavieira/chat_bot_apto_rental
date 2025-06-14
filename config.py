import os

from dotenv import load_dotenv

load_dotenv()

AI_CONTEXTUALIZE_PROMPT = os.getenv("AI_CONTEXTUALIZE_PROMPT")
AI_SYSTEM_PROMPT=os.getenv("AI_SYSTEM_PROMPT")
BUFFER_KEY_SUFIX = os.getenv("BUFFER_KEY_SUFIX")
BUFFER_TTL = os.getenv("BUFFER_TTL")
DEBOUNCE_SECONDS = os.getenv("DEBOUNCE_SECONDS")
EVOLUTION_API_URL = os.getenv("EVOLUTION_API_URL")
EVOLUTION_AUTHENTICATION_API_KEY = os.getenv("AUTHENTICATION_API_KEY")
EVOLUTION_INSTANCE_NAME = os.getenv("EVOLUTION_INSTANCE_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME")
OPENAI_MODEL_TEMPERATURE = os.getenv("OPENAI_MODEL_TEMPERATURE")
RAG_FILES_DIR = os.getenv("RAG_FILES_DIR")
REDIS_URL = os.getenv("CACHE_REDIS_URI")
VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH")

# Dados de login do painel administrativo de imóveis
REAL_ESTATE_ADMIN_URL = os.getenv("REAL_ESTATE_ADMIN_URL")
REAL_ESTATE_ADMIN_USERNAME = os.getenv("REAL_ESTATE_ADMIN_USERNAME")
REAL_ESTATE_ADMIN_PASSWORD = os.getenv("REAL_ESTATE_ADMIN_PASSWORD")

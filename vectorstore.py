
import os
import shutil
import logging
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config import RAG_FILES_DIR, VECTOR_STORE_PATH

# Configuração do logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_vectorstore() -> Chroma:
    """
    Carrega a vectorstore persistida do disco.
    Se não existir, cria uma vazia.
    """
    return Chroma(
        persist_directory=VECTOR_STORE_PATH,
        embedding_function=OpenAIEmbeddings(),
    )

def load_and_process_local_files():
    """
    Carrega documentos de arquivos locais (PDF, TXT) e os adiciona à vectorstore.
    Move os arquivos processados para uma subpasta para evitar reprocessamento.
    """
    docs = []
    processed_dir = os.path.join(RAG_FILES_DIR, 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    files_to_process = [
        os.path.join(RAG_FILES_DIR, f)
        for f in os.listdir(RAG_FILES_DIR)
        if f.endswith(('.pdf', '.txt'))
    ]

    if not files_to_process:
        logging.info("Nenhum novo arquivo local para processar.")
        return

    logging.info(f"Encontrados {len(files_to_process)} arquivos locais para processar.")

    for file_path in files_to_process:
        try:
            if file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
            else:
                loader = TextLoader(file_path)
            docs.extend(loader.load())
            
            # Move o arquivo para a pasta 'processed'
            shutil.move(file_path, os.path.join(processed_dir, os.path.basename(file_path)))
            logging.info(f"Arquivo '{os.path.basename(file_path)}' processado e movido.")
        except Exception as e:
            logging.error(f"Falha ao processar o arquivo {file_path}: {e}")

    if docs:
        add_documents_to_vectorstore(docs)

def add_documents_to_vectorstore(documents: list[Document]):
    """
    Adiciona uma lista de documentos à vectorstore.
    """
    if not documents:
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    
    logging.info(f"{len(chunks)} novos pedaços de documentos adicionados à vectorstore.")

def clear_vectorstore_by_source(source_prefix: str):
    """
    (Opcional/Avançado) Remove documentos da vectorstore baseados em uma fonte.
    A API do Chroma para remoção é limitada, a abordagem mais segura é recriar.
    Esta função é um placeholder para uma lógica de limpeza mais robusta.
    Por enquanto, vamos recriar o banco ao invés de limpar seletivamente.
    """
    logging.warning(f"A limpeza seletiva por fonte ('{source_prefix}') não é totalmente suportada. A recriação é mais segura.")
    # Em uma implementação real, você poderia buscar por IDs e deletá-los.
    # Ex: ids = vectorstore.get(where={"source": {"$like": f"{source_prefix}%"}})["ids"]
    # if ids: vectorstore.delete(ids)

def rebuild_vectorstore(documents: list[Document]):
    """
    Apaga a vectorstore antiga e a reconstrói com novos documentos.
    Útil para o scraping, para garantir que apenas imóveis atuais existam.
    """
    # Apaga o diretório antigo
    if os.path.exists(VECTOR_STORE_PATH):
        shutil.rmtree(VECTOR_STORE_PATH)
    
    os.makedirs(VECTOR_STORE_PATH)

    logging.info(f"Vectorstore antiga removida. Reconstruindo com {len(documents)} novos documentos.")
    add_documents_to_vectorstore(documents)


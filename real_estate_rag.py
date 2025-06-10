import threading
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from config import (
    VECTOR_STORE_PATH,
    REAL_ESTATE_ADMIN_URL,
    REAL_ESTATE_ADMIN_USERNAME,
    REAL_ESTATE_ADMIN_PASSWORD,
)

SCRAPE_INTERVAL_SECONDS = 30 * 60  # 30 minutos (1800 segundos)

def log(*args):
    print('[RAG]', *args)

def login_admin(session: requests.Session) -> bool:
    """Realiza login no painel administrativo."""
    if not all(
        [REAL_ESTATE_ADMIN_URL, REAL_ESTATE_ADMIN_USERNAME, REAL_ESTATE_ADMIN_PASSWORD]
    ):
        log("Variáveis de ambiente para login não configuradas. Pulando scraping.")
        return False

    login_url = urljoin(REAL_ESTATE_ADMIN_URL, "login")
    data = {
        "username": REAL_ESTATE_ADMIN_USERNAME,
        "password": REAL_ESTATE_ADMIN_PASSWORD,
    }

    try:
        response = session.post(login_url, data=data)
        response.raise_for_status()
        log("Login realizado com sucesso no painel de imóveis")
        return True
    except Exception as e:
        log(f"Falha ao realizar login: {e}")
        return False


def scrape_real_estate_site():
    """Faz scraping do painel admin e retorna os imóveis disponíveis."""
    session = requests.Session()
    if not login_admin(session):
        return []

    try:
        response = session.get(REAL_ESTATE_ADMIN_URL)
        response.raise_for_status()
    except Exception as e:
        log(f"Erro ao acessar página inicial: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    tab_links = [urljoin(REAL_ESTATE_ADMIN_URL, a["href"]) for a in soup.find_all("a", href=True) if "apto" in a["href"]]

    documents = []
    for link in tab_links:
        try:
            page = session.get(link)
            page.raise_for_status()
        except Exception as e:
            log(f"Falha ao acessar {link}: {e}")
            continue

        page_soup = BeautifulSoup(page.text, "html.parser")
        rows = page_soup.find_all("tr")
        log(f"[{link}] {len(rows)} linhas encontradas")

        for row in rows:
            cells = row.find_all("td")
            if not cells:
                continue

            # Verifica coluna de disponibilidade
            row_text = " ".join(cell.get_text(" ", strip=True).lower() for cell in cells)
            available = any(word in row_text for word in ["disponivel", "available", "sim", "yes", "true", "ativo"])
            if not available:
                continue

            details = [cell.get_text(" ", strip=True) for cell in cells]
            img = row.find("img")
            if img and img.get("src"):
                details.append(f"Foto: {img.get('src')}")

            content = "\n".join(filter(None, details))
            documents.append(Document(page_content=content))

    log(f"{len(documents)} documentos coletados do painel")
    return documents


def get_real_estate_vectorstore():
    documents = scrape_real_estate_site()

    if not documents:
        return Chroma(
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=OpenAIEmbeddings()
        )

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    return Chroma.from_documents(
        documents=chunks,
        embedding=OpenAIEmbeddings(),
        persist_directory=VECTOR_STORE_PATH,
    )


def refresh_real_estate_vectorstore():
    print("[INFO] Atualizando vetor de imóveis...")
    documents = scrape_real_estate_site()

    if documents:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        # recria o vetor sobrescrevendo o anterior
        Chroma.from_documents(
            documents=chunks,
            embedding=OpenAIEmbeddings(),
            persist_directory=VECTOR_STORE_PATH,
        )
        print(f"[INFO] Vetor de imóveis atualizado com {len(chunks)} documentos.")
    else:
        print("[WARN] Nenhum imóvel encontrado para atualizar vetor.")

    # agenda a próxima atualização
    threading.Timer(SCRAPE_INTERVAL_SECONDS, refresh_real_estate_vectorstore).start()


def start_real_estate_scheduler():
    print("[INFO] Iniciando atualização automática do RAG de imóveis...")
    refresh_real_estate_vectorstore()

def verify_real_estate_vectorstore():
    try:
        # Carrega o vetor de armazenamento existente
        vectorstore = Chroma(
            persist_directory=VECTOR_STORE_PATH,
            embedding_function=OpenAIEmbeddings()
        )

        # Recupera os documentos armazenados
        retriever = vectorstore.as_retriever()
        docs = retriever.get_relevant_documents("imóveis disponíveis")

        print(f"[INFO] Total de documentos armazenados: {len(docs)}")
        for i, doc in enumerate(docs, start=1):
            print(f"[Document {i}]")
            print(doc.page_content)
            print("-" * 50)

    except Exception as e:
        print(f"[ERROR] Falha ao verificar o vetor de armazenamento: {e}")

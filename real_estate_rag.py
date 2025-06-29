import logging
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from config (
    REAL_ESTATE_ADMIN_URL,
    REAL_ESTATE_ADMIN_USERNAME,
    REAL_ESTATE_ADMIN_PASSWORD,
)
from vectorstore import rebuild_vectorstore, load_and_process_local_files, get_vectorstore

# Configuração do logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def login_admin(session: requests.Session) -> bool:
    """Realiza login no painel administrativo."""
    if not all([REAL_ESTATE_ADMIN_URL, REAL_ESTATE_ADMIN_USERNAME, REAL_ESTATE_ADMIN_PASSWORD]):
        logging.warning("Variáveis de ambiente para login não configuradas. Pulando scraping.")
        return False

    login_url = urljoin(REAL_ESTATE_ADMIN_URL, "login")
    data = {"username": REAL_ESTATE_ADMIN_USERNAME, "password": REAL_ESTATE_ADMIN_PASSWORD}

    try:
        response = session.post(login_url, data=data)
        response.raise_for_status()
        logging.info("Login realizado com sucesso no painel de imóveis")
        return True
    except requests.RequestException as e:
        logging.error(f"Falha ao realizar login: {e}")
        return False

def scrape_real_estate_site() -> list[Document]:
    """Faz scraping do painel admin e retorna os imóveis disponíveis como Documentos."""
    session = requests.Session()
    if not login_admin(session):
        return []

    try:
        response = session.get(REAL_ESTATE_ADMIN_URL)
        response.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"Erro ao acessar página inicial do admin: {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    # Encontra todos os links que contêm 'apto' para navegar para as páginas de apartamentos
    tab_links = [urljoin(REAL_ESTATE_ADMIN_URL, a["href"]) for a in soup.find_all("a", href=True) if "apto" in a["href"]]

    documents = []
    for link in tab_links:
        try:
            page_response = session.get(link)
            page_response.raise_for_status()
        except requests.RequestException as e:
            logging.error(f"Falha ao acessar a página de apartamentos {link}: {e}")
            continue

        page_soup = BeautifulSoup(page_response.text, "html.parser")
        rows = page_soup.find_all("tr")
        logging.info(f"Analisando {len(rows)} imóveis em {link}")

        for row in rows:
            cells = row.find_all("td")
            if not cells:
                continue

            # Heurística para verificar se o imóvel está disponível
            row_text = " ".join(cell.get_text(" ", strip=True).lower() for cell in cells)
            is_available = any(word in row_text for word in ["disponivel", "available", "sim", "yes", "true", "ativo"])
            
            if not is_available:
                continue

            details = [cell.get_text(" ", strip=True) for cell in cells]
            img_tag = row.find("img")
            if img_tag and img_tag.get("src"):
                details.append(f"Foto: {img_tag.get('src')}")

            # Cria um Documento para cada imóvel disponível
            content = "\n".join(filter(None, details))
            # Adiciona metadados para futura identificação
            metadata = {"source": "real_estate_scrape", "url": link}
            documents.append(Document(page_content=content, metadata=metadata))

    logging.info(f"{len(documents)} imóveis disponíveis coletados do painel.")
    return documents

def refresh_knowledge_base():
    """
    Job completo de atualização da base de conhecimento.
    1. Faz o scraping dos imóveis.
    2. Reconstrói a vectorstore com os dados do scraping (para remover imóveis antigos).
    3. Carrega e adiciona quaisquer novos arquivos locais (PDFs, TXTs).
    """
    logging.info("Iniciando job de atualização da base de conhecimento...")
    
    # Passo 1: Scraping
    scraped_docs = scrape_real_estate_site()
    
    # Passo 2: Recria a base com os dados do scraping
    if scraped_docs:
        rebuild_vectorstore(scraped_docs)
    else:
        logging.warning("Nenhum documento retornado do scraping. A base de conhecimento de imóveis pode estar vazia.")

    # Passo 3: Adiciona arquivos locais
    # Isso garante que os arquivos PDF/TXT sejam adicionados após a reconstrução.
    load_and_process_local_files()
    
    logging.info("Job de atualização da base de conhecimento concluído.")

def verify_vectorstore_content():
    """
    Função de depuração para verificar o conteúdo atual da vectorstore.
    """
    try:
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        docs = retriever.get_relevant_documents("quais são os imóveis disponíveis?")

        logging.info(f"[VERIFICAÇÃO] Total de documentos recuperados: {len(docs)}")
        for i, doc in enumerate(docs, start=1):
            logging.info(f"[Documento {i}] Conteúdo: {doc.page_content[:300]}... | Metadados: {doc.metadata}")
        logging.info("-" * 50)

    except Exception as e:
        logging.error(f"[VERIFICAÇÃO] Falha ao verificar a vectorstore: {e}")
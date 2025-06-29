
import logging
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from config import WEB_PAGE_URL
from vectorstore import rebuild_vectorstore, load_and_process_local_files, get_vectorstore

# Configuração do logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def scrape_public_real_estate_site() -> list[Document]:
    """Faz scraping de um site público de imóveis e retorna os imóveis como Documentos."""
    if not WEB_PAGE_URL:
        logging.warning("A variável de ambiente WEB_PAGE_URL não está configurada. Pulando scraping.")
        return []

    try:
        response = requests.get(WEB_PAGE_URL, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        logging.info(f"Acessando o site: {WEB_PAGE_URL}")
    except requests.RequestException as e:
        logging.error(f"Erro ao acessar a URL pública {WEB_PAGE_URL}: {e}")
        return []

    soup = BeautifulSoup(response.content, 'html.parser')
    documents = []

    # --- ÁREA DE CUSTOMIZAÇÃO --- 
    # Você PRECISA ajustar estes seletores para o site alvo.
    # Exemplo genérico:
    listings = soup.find_all('div', class_='imovel-card') # Mude 'div' e 'imovel-card'
    logging.info(f"Encontrados {len(listings)} anúncios de imóveis no site.")

    for apt in listings:
        try:
            # Extraia os dados. Mude os seletores conforme necessário.
            title = apt.find('h2', class_='imovel-titulo').text.strip()
            address = apt.find('p', class_='imovel-endereco').text.strip()
            price = apt.find('p', class_='imovel-preco').text.strip()
            link_tag = apt.find('a')
            link = link_tag['href'] if link_tag else "Link não encontrado"

            # Monta uma descrição para ser usada pelo LLM
            content = (
                f"Imóvel: {title}. "
                f"Localização: {address}. "
                f"Preço: {price}. "
                f"Link para mais detalhes: {link}"
            )
            
            metadata = {"source": "web_scrape", "url": link}
            documents.append(Document(page_content=content, metadata=metadata))
        except AttributeError as e:
            logging.warning(f"Não foi possível extrair todos os dados de um anúncio. Erro: {e}")
            continue
            
    logging.info(f"{len(documents)} imóveis coletados do site público.")
    return documents

def refresh_knowledge_base():
    """
    Job completo de atualização da base de conhecimento.
    1. Faz o scraping dos imóveis do site público.
    2. Reconstrói a vectorstore com os dados do scraping.
    3. Carrega e adiciona quaisquer novos arquivos locais (PDFs, TXTs).
    """
    logging.info("Iniciando job de atualização da base de conhecimento...")
    
    scraped_docs = scrape_public_real_estate_site()
    
    if scraped_docs:
        rebuild_vectorstore(scraped_docs)
    else:
        logging.warning("Nenhum documento retornado do scraping. A base de conhecimento de imóveis pode estar vazia.")

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
        if not docs:
            logging.info("[VERIFICAÇÃO] A vectorstore parece estar vazia.")
            return
            
        for i, doc in enumerate(docs, start=1):
            logging.info(f"[Documento {i}] Conteúdo: {doc.page_content[:300]}... | Metadados: {doc.metadata}")
        logging.info("-" * 50)

    except Exception as e:
        logging.error(f"[VERIFICAÇÃO] Falha ao verificar a vectorstore: {e}")

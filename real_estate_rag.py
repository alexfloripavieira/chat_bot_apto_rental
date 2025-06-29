
import logging
from urllib.parse import urljoin
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

    # Encontra a tabela principal que contém os imóveis
    table = soup.find('table')
    if not table:
        logging.warning("Nenhuma tabela de imóveis encontrada no site.")
        return []

    # Pega todas as linhas do corpo da tabela, ignorando o cabeçalho e rodapé
    rows = table.find('tbody').find_all('tr')
    logging.info(f"Encontradas {len(rows)} linhas de imóveis na tabela.")

    for row in rows:
        # Pega todas as células da linha
        cells = row.find_all('td')
        if len(cells) < 6:  # Garante que a linha tem todas as colunas esperadas
            continue

        try:
            # Extrai os dados de cada célula pela ordem
            local = cells[0].text.strip()
            numero = cells[1].text.strip()
            andar = cells[2].text.strip()
            valor = cells[3].text.strip()
            mobiliado = cells[4].text.strip()
            
            # Encontra o link na última célula
            link_tag = cells[5].find('a')
            if link_tag and link_tag.get('href'):
                # Constrói a URL completa do link
                relative_link = link_tag['href']
                full_link = urljoin(WEB_PAGE_URL, relative_link)
            else:
                full_link = "Link não disponível"

            # Monta uma descrição clara para ser usada pelo LLM
            content = (
                f"Imóvel disponível no {local}, número {numero} ({andar}).\n"
                f"Valor do aluguel: {valor}.\n"
                f"Mobiliado: {mobiliado}.\n"
                f"Para mais detalhes, acesse o link."
            )
            
            metadata = {"source": "web_scrape", "url": full_link, "local": local}
            documents.append(Document(page_content=content, metadata=metadata))
        except IndexError as e:
            logging.warning(f"Não foi possível extrair dados de uma linha da tabela. Erro: {e}")
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

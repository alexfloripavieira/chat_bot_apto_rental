import threading
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from config import VECTOR_STORE_PATH

SCRAPE_INTERVAL_SECONDS = 30  # 30 minutos

def log(*args):
    print('[RAG]', *args)

def scrape_real_estate_site():
    url = "https://www.robsonvieira.com.br/imoveis"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")

    # Seleciona as linhas da tabela que contêm os dados dos imóveis
    property_rows = soup.select("div#main table tr.altrow")
    log(f"Encontradas {len(property_rows)} linhas de imóveis.")

    documents = []
    for row in property_rows:
        # Extrai os dados de cada célula da linha
        cells = row.find_all("td")
        if len(cells) >= 5:  # Certifica-se de que há pelo menos 5 colunas
            location = cells[0].get_text(strip=True)
            number = cells[1].get_text(strip=True)
            floor = cells[2].get_text(strip=True)
            price = cells[3].get_text(strip=True)
            furnished = cells[4].get_text(strip=True)

            # Monta o conteúdo do documento
            content = "\n".join(
                filter(None, [
                    f"Localização: {location}",
                    f"Número: {number}",
                    f"Andar: {floor}",
                    f"Valor: {price}",
                    f"Mobiliado: {furnished}",
                ])
            )

            # Adiciona o documento à lista
            documents.append(Document(page_content=content))

    log(f"documents {documents}")

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

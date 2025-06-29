
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from chains import get_conversational_rag_chain
from evolution_api import send_whatsapp_message
from real_estate_rag import refresh_knowledge_base, verify_vectorstore_content

# Configuração do logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Job de atualização da base de conhecimento a cada 30 minutos
SCRAPE_INTERVAL_MINUTES = 30

# Cria o scheduler que rodará em background
scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Inicia a base de conhecimento na inicialização da aplicação
    logging.info("Executando a primeira atualização da base de conhecimento...")
    refresh_knowledge_base()
    
    # Agenda a tarefa para execuções futuras
    scheduler.add_job(refresh_knowledge_base, 'interval', minutes=SCRAPE_INTERVAL_MINUTES)
    scheduler.start()
    logging.info(f"Agendador configurado para atualizar a base a cada {SCRAPE_INTERVAL_MINUTES} minutos.")
    
    yield
    
    # Desliga o scheduler quando a aplicação parar
    scheduler.shutdown()

app = FastAPI(lifespan=lifespan)

# Carrega a cadeia de conversação uma vez na inicialização
conversational_rag_chain = get_conversational_rag_chain()

@app.post("/webhook")
async def webhook(request: Request):
    try:
        data = await request.json()
        
        # Extração segura dos dados da mensagem
        message_data = data.get("data", {})
        if not message_data:
            return {"status": "ok", "reason": "no data"}

        chat_id = message_data.get("key", {}).get("remoteJid")
        message = message_data.get("message", {}).get("conversation")
        from_me = message_data.get("key", {}).get("fromMe", False)

        # Ignora mensagens de grupo, de status ou enviadas por mim mesmo
        if not all([chat_id, message]) or "@g.us" in chat_id or from_me:
            return {"status": "ok", "reason": "ignored"}

        logging.info(f"Mensagem recebida de {chat_id}: {message}")

        # Invoca a cadeia de RAG com o histórico da sessão
        response = conversational_rag_chain.invoke(
            {"input": message},
            config={"configurable": {"session_id": chat_id}},
        )

        answer = response.get("answer", "Desculpe, não consegui processar sua pergunta.")
        
        # Envia a resposta para o usuário
        send_whatsapp_message(chat_id, answer)
        logging.info(f"Resposta enviada para {chat_id}: {answer}")

    except Exception as e:
        logging.error(f"Erro fatal no processamento do webhook: {e}")
        # Retorna um erro HTTP para que o serviço de webhook possa registrar a falha
        raise HTTPException(status_code=500, detail="Internal Server Error")

    return {"status": "sent"}


@app.get("/verify-vectorstore")
async def verify_vectorstore_endpoint():
    """
    Endpoint para acionar manualmente a verificação do conteúdo da vectorstore.
    """
    logging.info("Verificação manual da vectorstore acionada via endpoint.")
    verify_vectorstore_content()
    return {"status": "ok", "message": "Verificação concluída. Veja os logs do servidor para detalhes."}


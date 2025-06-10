from fastapi import FastAPI, Request
import logging

from chains import get_conversational_rag_chain
from message_buffer import buffer_message
from real_estate_rag import start_real_estate_scheduler, verify_real_estate_vectorstore

# Configuração do logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI()

conversational_rag_chain = get_conversational_rag_chain()

# Inicia o agendador para atualizar o vetor de armazenamento de imóveis
start_real_estate_scheduler()


@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    chat_id = data.get("data").get("key").get("remoteJid")
    message = data.get("data").get("message").get("conversation")

    if chat_id and message and "@g.us" not in chat_id:
        await buffer_message(
            chat_id=chat_id,
            message=message,
        )
    return {"status": "ok"}


@app.get("/verify-vectorstore")
async def verify_vectorstore():
    """
    Endpoint para verificar os dados armazenados no vetor de imóveis.
    """
    verify_real_estate_vectorstore()
    return {
        "status": "ok",
        "message": "Verificação concluída. Veja os logs para detalhes.",
    }

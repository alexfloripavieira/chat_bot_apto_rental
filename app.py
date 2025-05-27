from fastapi import FastAPI, Request
import logging

from chains import get_conversational_rag_chain

from message_buffer import buffer_message

# Configuração do logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

app = FastAPI()

convertional_rag_chain = get_conversational_rag_chain()


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

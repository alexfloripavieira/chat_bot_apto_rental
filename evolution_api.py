import requests

from config import AUTHENTICATION_API_KEY, EVOLUTION_API_URL, EVOLUTION_INSTANCE_NAME


def send_whatsapp_message(number, text):
    url = f"{EVOLUTION_API_URL}/message/sendText/{EVOLUTION_INSTANCE_NAME}"
    headers = {"apikey": AUTHENTICATION_API_KEY, "Content-Type": "application/json"}
    payload = {"number": number, "text": text}
    requests.post(url=url, headers=headers, json=payload)

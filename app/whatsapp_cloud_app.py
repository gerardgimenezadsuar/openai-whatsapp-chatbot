import os
import logging
from datetime import datetime
from dotenv import find_dotenv, load_dotenv
from flask import Flask, request, jsonify
from chat.clients.whatsapp_cloud import WhatsAppCloudClient, parse_whatsapp_message
from chat.handlers.openai import (
    chat_completion as chatgpt_completion,
    voice_transcription as whisper_transcription,
)
from app.handlers import (
    check_and_send_image_generation,
    ensure_user_language,
    verify_image_generation,
    verify_and_process_media,
    check_conversation_end,
)
from app.whatsapp.chat import Sender, OpenAIChatManager

# Load environment variables
load_dotenv(find_dotenv())
logging.basicConfig()
logger = logging.getLogger("WP-CLOUD-APP")
logger.setLevel(logging.DEBUG)

# Initialize WhatsApp client
whatsapp_client = WhatsAppCloudClient(
    phone_number_id=os.environ.get("WHATSAPP_PHONE_NUMBER_ID"),
    access_token=os.environ.get("WHATSAPP_ACCESS_TOKEN"),
)

# Chat configuration
start_template = os.environ.get("CHAT_START_TEMPLATE")
if start_template and os.path.exists(start_template):
    with open(start_template, "r") as f:
        start_template = f.read()

chat_options = dict(
    model=os.environ.get("CHAT_MODEL", "llama-3.3-70b-versatile"),
    agent_name=os.environ.get("AGENT_NAME", "ParityDx"),
    start_system_message=start_template,
    goodbye_message="Goodbye! I'll be here if you need me.",
    voice_transcription=True,
    allow_images=True,
)

model_options = dict(
    model=os.environ.get("CHAT_MODEL", "llama-3.3-70b-versatile"),
    max_tokens=int(os.environ.get("MAX_TOKENS", 400)),
    temperature=float(os.environ.get("TEMPERATURE", 0.7)),
    top_p=float(os.environ.get("TOP_P", 1)),
    frequency_penalty=float(os.environ.get("FREQUENCY_PENALTY", 0)),
    presence_penalty=float(os.environ.get("PRESENCE_PENALTY", 0)),
    n=1,
)

# Create Flask app
app = Flask(__name__)


@app.route("/webhook", methods=["GET"])
def verify_webhook():
    """Webhook verification endpoint for WhatsApp"""
    verify_token = os.environ.get("WHATSAPP_VERIFY_TOKEN", "your-verify-token")
    
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    
    if mode == "subscribe" and token == verify_token:
        logger.info("Webhook verified successfully")
        return challenge, 200
    else:
        logger.error("Webhook verification failed")
        return "Forbidden", 403


@app.route("/webhook", methods=["POST"])
async def process_webhook():
    """Process incoming WhatsApp messages"""
    try:
        data = request.get_json()
        logger.info(f"Received webhook data: {data}")
        
        # Parse the incoming message
        message_data = parse_whatsapp_message(data)
        if not message_data:
            return jsonify({"status": "ok"}), 200
            
        # Only process text messages for now
        if message_data["type"] != "text":
            return jsonify({"status": "ok"}), 200
            
        # Mark message as read
        whatsapp_client.mark_as_read(message_data["message_id"])
        
        # Create or get chat manager
        sender = Sender(
            phone_number=message_data["from"],
            name=message_data["name"],
        )
        chat = OpenAIChatManager.get_or_create(sender, logger=logger, **chat_options)
        
        # Set system message
        if chat_options.get("start_system_message"):
            chat.start_system_message = chat_options.get("start_system_message").format(
                user=sender.name, today=datetime.now().strftime("%Y-%m-%d")
            )
            if len(chat.messages) == 0 or chat.messages[0]["role"] != "system":
                chat.messages.insert(0, chat.make_message(chat.start_system_message, role="system"))
        
        # Check for language on first message
        if len(chat.messages) == 1:
            await ensure_user_language(chat, text=message_data["text"])
            
        # Add user message
        chat.add_message(message_data["text"], role="user")
        
        # Generate reply
        reply = chatgpt_completion(chat.messages, **model_options).strip()
        logger.info(f"Generated reply of length {len(reply)}")
        
        # Check for image generation request
        reply, img_prompt = verify_image_generation(reply)
        
        # Send reply (split if too long)
        if len(reply) > 1600:
            for i in range(0, len(reply), 1500):
                chunk = reply[i:i+1500]
                if i + 1500 < len(reply):
                    chunk += "..."
                whatsapp_client.send_message(
                    to=message_data["from"],
                    text=chunk
                )
        else:
            whatsapp_client.send_message(
                to=message_data["from"],
                text=reply
            )
            
        # Add assistant message to chat
        chat.add_message(reply, role="assistant")
        
        # Handle image generation if requested
        if img_prompt:
            try:
                image_url = await check_and_send_image_generation(
                    img_prompt, chat, whatsapp_client, message_data["from"]
                )
                if image_url:
                    whatsapp_client.send_image(
                        to=message_data["from"],
                        image_url=image_url,
                        caption=f"Generated image: {img_prompt}"
                    )
            except Exception as e:
                logger.error(f"Image generation failed: {e}")
                whatsapp_client.send_message(
                    to=message_data["from"],
                    text="Sorry, I couldn't generate the image. Please try again."
                )
        
        return jsonify({"status": "ok"}), 200
        
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        return jsonify({"status": "error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
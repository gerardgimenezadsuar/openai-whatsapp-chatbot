"""
WhatsApp Cloud API Client
"""
import os
import json
import logging
import requests
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class WhatsAppCloudClient:
    """Client for WhatsApp Cloud API"""
    
    def __init__(
        self,
        phone_number_id: str,
        access_token: str,
        api_version: str = "v21.0"
    ):
        self.phone_number_id = phone_number_id
        self.access_token = access_token
        self.api_version = api_version
        self.base_url = f"https://graph.facebook.com/{api_version}"
        
    def send_message(
        self,
        to: str,
        text: str,
        preview_url: bool = False
    ) -> Dict[str, Any]:
        """Send a text message"""
        url = f"{self.base_url}/{self.phone_number_id}/messages"
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        data = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "text",
            "text": {
                "preview_url": preview_url,
                "body": text
            }
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send message: {e}")
            raise
            
    def send_image(
        self,
        to: str,
        image_url: str,
        caption: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send an image message"""
        url = f"{self.base_url}/{self.phone_number_id}/messages"
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        data = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": "image",
            "image": {
                "link": image_url
            }
        }
        
        if caption:
            data["image"]["caption"] = caption
            
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to send image: {e}")
            raise
            
    def mark_as_read(self, message_id: str) -> Dict[str, Any]:
        """Mark a message as read"""
        url = f"{self.base_url}/{self.phone_number_id}/messages"
        
        headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
        
        data = {
            "messaging_product": "whatsapp",
            "status": "read",
            "message_id": message_id
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to mark message as read: {e}")
            raise


def parse_whatsapp_message(webhook_data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse incoming WhatsApp webhook data"""
    try:
        entry = webhook_data["entry"][0]
        changes = entry["changes"][0]
        value = changes["value"]
        
        if "messages" in value:
            message = value["messages"][0]
            contact = value["contacts"][0]
            
            return {
                "from": message["from"],
                "name": contact["profile"]["name"],
                "message_id": message["id"],
                "timestamp": message["timestamp"],
                "type": message["type"],
                "text": message.get("text", {}).get("body", "") if message["type"] == "text" else None,
                "image": message.get("image") if message["type"] == "image" else None,
                "audio": message.get("audio") if message["type"] == "audio" else None,
            }
    except (KeyError, IndexError) as e:
        logger.error(f"Failed to parse webhook data: {e}")
        return None
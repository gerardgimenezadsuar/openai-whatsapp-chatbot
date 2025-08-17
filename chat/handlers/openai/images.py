import os, logging
from chat.clients import ChatClient
from openai import OpenAI
import requests

async def text_to_image(prompt: str, *, as_url=True, **kwargs):
    """Generate an image asychronously given the prompt"""
    logging.debug(
        f"Querying OpenAI's Image API 'DALL-E' with prompt '{prompt}'"
    )
    creation_params = dict(n=1, size="1024x1024")
    creation_params.update(kwargs)
    
    # Initialize the OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    # create a partial function to send the image with the given parameters
    response = client.images.generate(
        prompt=prompt,
        model="dall-e-3",
        **creation_params)
    
    if not response.data:
        return None
    
    image_url = response.data[0].url
    if not image_url:
        return None
    if as_url:
        return image_url
    else:
        return requests.get(image_url).content
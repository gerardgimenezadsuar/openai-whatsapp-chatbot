import tempfile
import os
from typing import List, Union

from openai import OpenAI
import requests

from chat.clients import ChatClient

def voice_transcription(
    url_or_file: str,
    chat: ChatClient = None,
    *,
    language: str = 'en',
    model: str = 'whisper-1',
    prompt: str = None,
    asynch: bool = False,
    **kwargs
) -> str:
    """
    Transcribes the given audio file or URL using OpenAI's Voice API.
    Returns the transcription as a string.

    Parameters
    ----------
    url_or_file : str
        The URL or path to the audio file to be transcribed.
    chat : ChatClient, optional
        The chat client to use for logging, by default None
    **kwargs
    """
    if url_or_file.startswith("http"):
        response = requests.get(url_or_file)
        response.raise_for_status()
        audio = response.content
        # create the file
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(audio)
        url_or_file = f.name

    # Initialize the OpenAI client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    with open(url_or_file, 'rb') as audio_file:
        response = client.audio.transcriptions.create(
            file=audio_file,
            model=model,
            prompt=prompt,
            language=language,
            response_format='json',
            **kwargs
        )
    
    return response.text

def voice_translation(
    url_or_file: str,
    chat: ChatClient = None,
    *,
    language: str = 'en',
    model: str = 'whisper-1',
    prompt: str = None,
    **kwargs
) -> str:
    """
    Translates the given audio file or URL using OpenAI's Voice API.
    Returns the translation as a string.

    Parameters
    ----------
    url_or_file : str
        The URL or path to the audio file to be translated.
    chat : ChatClient, optional
        The chat client to use for logging, by default None
    **kwargs
    """
    if url_or_file.startswith("http"):
        response = requests.get(url_or_file)
        response.raise_for_status()
        audio = response.content
        # create the file
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(audio)
        url_or_file = f.name

    response = openai.Audio.translate(
        url_or_file, model=model, prompt=prompt, 
        language=language, response_format='json', 
        **kwargs)

    return response.get("text")
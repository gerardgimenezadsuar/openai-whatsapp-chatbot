"""
Handlers for OpenAI's Completion and Chat Completion APIs
"""
import logging, re
import os
from typing import List

from openai import OpenAI
from groq import Groq
from chat.clients import ChatClient

__all__ = [
    "text_completion",
    "chat_completion",
    "code_generation",
    "whisper_voice_transcription",
    "dalle_text_to_image",
]

def text_completion(
    prompt: str,
    chat: ChatClient = None,
    engine: str = "gpt-4o-mini",
    **kwargs
):
    """
    Generates text completion using OpenAI's Chat Completion API.
    Now uses chat completion models as text completion models are deprecated.

    Parameters
    ----------
    prompt : str
        The prompt to complete.
    chat : ChatClient, optional
        The chat client, by default None
    engine : str, optional
        The engine to use, by default "gpt-4o-mini"
    **kwargs
        Additional keyword arguments to pass to the Completion API.
    """
    if "model" in kwargs:
        engine = kwargs.pop("model")
    logging.info(f"Querying OpenAI's Chat Completion API with prompt '{prompt}'")
    
    # Convert prompt to messages format
    if isinstance(prompt, list):
        messages = prompt
    else:
        messages = [{"role": "user", "content": prompt}]
    
    return chat_completion(messages, model=engine, **kwargs)

def chat_completion(
    messages: List[dict],
    model: str = "gpt-4o-mini",
    **kwargs
):
    """
    Generates chat completion using OpenAI's or Groq's Chat Completion API.

    Parameters
    ----------
    messages : List[dict]
        A list of messages to complete.
    chat : ChatClient, optional
        The chat client, by default None
    model : str, optional
        The model to use, by default "gpt-4o-mini"
    **kwargs
        Additional keyword arguments to pass to the Chat Completion API.
        See https://platform.openai.com/docs/api-reference/chat/create for a list of
        valid parameters.
    """
    if "engine" in kwargs:
        model = kwargs.pop("engine")
    
    # Check if we should use Groq
    use_groq = os.environ.get("USE_GROQ", "false").lower() == "true"
    
    if use_groq:
        # Initialize the Groq client
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        # Default to llama model if not specified
        if model in ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4"]:
            model = "llama-3.3-70b-versatile"
    else:
        # Initialize the OpenAI client
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs
    )

    return response.choices[0].message.content


def text_translation(
    text: str,
    to: str = "english",
    from_: str = None,
    engine: str = "gpt-4o-mini",
    prompt: str = None,
    examples: List[str] = None,
    **kwargs
):
    """
    Translates text using OpenAI's Completion API.
    It injects a translation prompt into the text to be translated

    Parameters
    ----------
    text : str
        The text to translate.
    to : str, optional
        The language to translate to. By default "english".
    from_ : str, optional
        The language to translate from. By default the language is inferred from the
        text.
    engine : str, optional
        The engine to use, by default "text-davinci-003" for chat completion.
    prompt : str, optional
        The prompt to inject into the text to be translated.
    examples : List[Tuple], optional
        A list of few-show examples to use for the translation task in the form of
        (text, translation) tuples. e.g. [("Hello world", "Bonjour le monde")] for
        English to French translation.
    **kwargs
        Additional keyword arguments to pass to the Completion API.
    
    Returns
    -------
    str
        The translated text.
    
    Usage
    -----
    >>> translate_text("Hello world", to="french")
    "Bonjour le monde"
    >>> translate_text("Comment allez-vous?", from_="french", to="spanish", examples=[""Bonjour le monde", "Hola Mundo"])
    "¿Cómo estás?"
    """
    logging.info(f"Querying OpenAI's Completion API with prompt '{prompt}'")
    if prompt is None:
        if from_ is None:
            prompt = f"Translate the text '{text}' to {to.capitalize()}."
        else:
            prompt = f"Translate the text '{text}' from {from_.capitalize()} to {to.capitalize()}."
    if examples is not None:
        prompt += " For example: " + ", ".join(
            [f"{txt} -> {translation}" for txt, translation in examples]
        )
    prompt += f"\n----\n{text} ->"
    if engine == 'gpt-3.5-turbo':
        messages = [
            {'role': 'user', 'content': prompt},
        ]
        # print(f"Querying OpenAI's Chat Completion API with messages {messages}")
        result_text = chat_completion(messages, model=engine, **kwargs)
    else:
        # print(f"Querying OpenAI's Completion API with prompt '{prompt}'")
        kwargs['stop'] = '\n'
        result_text = text_completion(prompt, engine=engine, **kwargs)
    return re.sub(r" ->.*", "", result_text).strip()

async def atext_translation(*args, **kwargs):
    return text_translation(*args, **kwargs)

def language_detection(
    text: str,
    engine: str = "gpt-4o-mini",
    prompt: str = None,
    examples: List[str] = None,
    **kwargs
) -> str:
    """ 
    Recognizes the language of a text using OpenAI's Completion API.
    It injects a language recognition prompt into the text to be recognized

    Parameters
    ----------
    text : str
        The text to recognize.
    engine : str, optional
        The engine to use, by default "gpt-3.5-turbo" for chat completion.
    prompt : str, optional
        The prompt to inject into the text to be recognized.
    examples : List[Tuple], optional
        A list of few-show examples to use for the language recognition task in the
        form of (text, language) tuples. e.g. [("Hello world", "english")]
    **kwargs
        Additional keyword arguments to pass to the Completion API.
    """
    if prompt is None:
        prompt = f"You are a language recognition program. You can only output a single word saying the language of a given text."
    else:
        prompt = prompt.format(text=text)
    if examples is None:
        examples = [
            ("Hello world", "english"),
            ("Bonjour le monde", "french"),
            ("Hola mundo", "spanish"),
            ("Hallo Welt", "german"),
        ]
    prompt += " Some \"text\" -> reply example outputs are: " + ", ".join(
        [f"\"{txt}\" -> {language}" for txt, language in examples]
    )
    prompt += f"\n---\n{text} ->"
    if engine == 'gpt-3.5-turbo':
        messages = [
            {'role': 'system', 'content': prompt},
        ]
        # kwargs['stop'] = ['\n']
        print(f"Querying OpenAI's Chat Completion API with messages {messages}")
        result_text = chat_completion(messages, model=engine, **kwargs)
    else:
        # print(f"Querying OpenAI's Completion API with prompt '{prompt}'")
        kwargs['stop'] = '\n'
        result_text = text_completion(prompt, engine=engine, **kwargs)
    detected_lang = re.sub(r" ->.*", "", result_text).strip().lower()
    if len(detected_lang.split()) > 1:
        detected_lang = detected_lang.split()[0]
    return detected_lang

async def alanguage_detection(*args, **kwargs):
    return language_detection(*args, **kwargs)

def code_generation(
    prompt: str,
    chat: ChatClient = None,
    engine: str = "gpt-4o-mini",
    **kwargs
):
    """
    Generates code completion using OpenAI's Chat Completion API.
    """
    logging.info(f"Querying OpenAI's Chat Completion API with prompt '{prompt}'")
    messages = [{"role": "user", "content": prompt}]
    return chat_completion(messages, model=engine, **kwargs)
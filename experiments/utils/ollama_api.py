import config
import openai
from openai import OpenAI
import os
import re
from time import sleep

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

# Set OpenAPI key from environment or config file
client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama', # required, but unused
)

# Define retry decorator to handle OpenAI API timeouts
@retry(wait=wait_random_exponential(min=20, max=100), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.completions.create(**kwargs)

# Define retry decorator to handle OpenAI API timeouts
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chat_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)


# Define ollama chat function
def OllamaChat(messages, model="wrn", num_samples=1):
    response = chat_with_backoff(
        model=model,
        messages=messages,
        temperature=0,
        top_p=1,
        max_tokens=512,
        n=num_samples
    )
    candidates = []
    for candidate in response.choices:
        z = candidate.message.content
        pred = re.sub("\n"," ", z)
        candidates.append(pred.strip())
    return candidates
    
def OllamaCompletion(messages, stop, model="wrn", num_samples=1):
    response = completion_with_backoff(
        model=model,
        messages=messages,
        temperature=0,
        top_p=1,
        max_tokens=512,
        n=num_samples,
        stop=stop
    )
    candidates = []
    for candidate in response.choices:
        z = candidate.message.content
        pred = re.sub("\n"," ", z)
        candidates.append(pred.strip())
    return candidates

if __name__ == "__main__":
    pass
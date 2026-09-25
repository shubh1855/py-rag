import os

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()

api_key = os.environ.get("OPENROUTER_API_KEY")

if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

messages: list[ChatCompletionMessageParam] = [
    {
        "role": "user",
        "content": "What is a RAG? Use one paragraph at maximum.",
    }
]

response = client.chat.completions.create(
    model="openrouter/free",
    messages=messages,
    extra_body={
        "usage": {
            "include": True,
        }
    },
)

print(response.choices[0].message.content)

if response.usage is None:
    raise RuntimeError("No usage information returned")

usage = response.usage

print(f"Prompt tokens: {usage.prompt_tokens}")
print(f"Response tokens: {usage.completion_tokens}")

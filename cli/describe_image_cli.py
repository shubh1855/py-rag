import argparse
import base64
import mimetypes
import os

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rewrite a movie search query using an image and text"
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the image file",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="Text query to rewrite based on the image",
    )
    args = parser.parse_args()

    # Determine the image MIME type.
    mime, _ = mimetypes.guess_type(args.image)
    mime = mime or "image/jpeg"

    # Read the image in binary mode.
    with open(args.image, "rb") as image_file:
        img = image_file.read()

    # Configure the OpenRouter client.
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        parser.error("OPENROUTER_API_KEY environment variable not set")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    system_prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""

    # Encode the image as a base64 data URL.
    data_url = f"data:{mime};base64,{base64.b64encode(img).decode()}"

    # Build the multimodal message.
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt.strip(),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": data_url,
                    },
                },
                {
                    "type": "text",
                    "text": args.query.strip(),
                },
            ],
        }
    ]

    # Send the request to OpenRouter.
    response = client.chat.completions.create(
        model="openrouter/free",
        messages=messages,
    )

    if not response.choices:
        raise RuntimeError("The model returned no response choices.")

    content = response.choices[0].message.content

    if not content or not content.strip():
        raise RuntimeError("The model returned an empty rewritten query.")

    print(f"Rewritten query: {content.strip()}")

    if response.usage is not None:
        print(f"Total tokens:    {response.usage.total_tokens}")


if __name__ == "__main__":
    main()

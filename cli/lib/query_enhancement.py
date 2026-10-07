import os

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

if not api_key:
    raise RuntimeError("OPENROUTER_API_KEY environment variable not set")


client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
)

model = "openrouter/free"


def _call_llm(prompt: str) -> str:
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    result = response.choices[0].message.content

    if not result:
        return ""

    return result.strip()


def spell_query(query: str) -> str:
    prompt = f"""Fix any spelling errors in the user-provided movie search query below.
Correct only clear, high-confidence typos. Do not rewrite, add, remove, or reorder words.
Preserve punctuation and capitalization unless a change is required for a typo fix.
If there are no spelling errors, or if you're unsure, output the original query unchanged.
Output only the final query text, nothing else.
User query: "{query}"
"""

    enhanced_query = _call_llm(prompt)

    return enhanced_query or query


def rewrite_query(query: str) -> str:
    prompt = f"""Rewrite the user-provided movie search query below to be more specific and searchable.

Consider:
- Common movie knowledge (famous actors, popular films)
- Genre conventions (horror = scary, animation = cartoon)
- Keep the rewritten query concise (under 10 words)
- It should be a Google-style search query, specific enough to yield relevant results
- Don't use boolean logic

Examples:
- "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
- "movie about bear in london with marmalade" -> "Paddington London marmalade"
- "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

If you cannot improve the query, output the original unchanged.
Output only the rewritten query text, nothing else.

User query: "{query}"
"""

    enhanced_query = _call_llm(prompt)

    return enhanced_query or query


def expand_query(query: str) -> str:
    prompt = f"""Expand the user-provided movie search query with related terms.

Add synonyms and related concepts that might appear in movie descriptions.
Keep expansions relevant and focused.
Prefer both subject-specific terms and concepts describing characters, themes, or situations.
Output only the additional terms; they will be appended to the original query.

Examples:
- "scary bear movie" -> "horror grizzly terrifying creature attack wilderness"
- "action movie with bear" -> "thriller chase fight survival wilderness"
- "comedy with bear" -> "funny humor lighthearted animal comedy"
- "math movie" -> "mathematics equations genius intelligence professor problem-solving"

User query: "{query}"
"""

    additional_terms = _call_llm(prompt)

    if not additional_terms:
        return query

    return f"{query} {additional_terms}"


def enhance_query(
    query: str,
    method: str,
) -> str:
    if method == "spell":
        return spell_query(query)

    if method == "rewrite":
        return rewrite_query(query)

    if method == "expand":
        return expand_query(query)

    return query

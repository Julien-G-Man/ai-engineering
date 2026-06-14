import os
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from typing import List, Dict, Optional

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_response(system_prompt, user_prompt):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    return completion.choices[0].message.content


def build_batched_messages(
    countries: List[str],
    system_prompt: Optional[str] = None,
    system_role: str = "system",
    user_role: str = "user",
) -> List[Dict[str, str]]:
    """
    Build a list of messages for an LLM call given a list of countries.

    - countries: list of country names to include as separate user messages.
    - system_prompt: optional system prompt; a sensible default is used when None.
    - returns: list of message dicts with 'role' and 'content'.
    """
    if system_prompt is None:
        system_prompt = (
            "You are given a series of countries and are asked to return the country "
            "and capital city. Provide each of the questions with an answer in the response "
            "as separate content."
        )

    messages: List[Dict[str, str]] = [{"role": system_role, "content": system_prompt}]
    for country in countries:
        messages.append({"role": user_role, "content": country})
    return messages


countries = ["United States", "Ireland", "India"]
messages = build_batched_messages(countries)


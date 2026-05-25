"""
Function calling with external API
"""

import os
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set")

client = OpenAI(api_key=api_key)

function_definition = [{
    'type': 'function',
    'function': {
        'name': 'get_artwork',
        'description': 'This function calls the Arts Institute of Chicago API to find artwork that matches a keyword',
        'parameters': {
            'type': 'object',
            'properties': {
                'artwork_keyword': {
                    'type': 'string',
                    'description': 'The keyword to be passed to the get_artwork function.'
                }
            }
        },
        'result_type': {'type': 'string'}
    }
}]


def get_artwork(keyword):
    url = "https://api.artic.edu/api/v1/artworks/search"
    querystring = {'q': keyword}
    response = requests.request("GET", url, params=querystring)
    return response.text


def get_response(user_prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
                    You are an AI assistant, a specialist in history.
                    You should interpret the user prompt, and based on it 
                    extract one keyword for recommending artwork related to their preference."""
            },
            {"role": "user", "content": user_prompt},
        ],
        tools=function_definition,
        tool_choice={'type': 'function', 
                     'function': {'name': 'get_artwork'}}
    )
    return get_recommendation(response)


def get_recommendation(response):
    try:
        # expect SDK to expose tool_calls as a list-like on the first choice
        call = response.choices[0].message.tool_calls[0]
        func = call['function'] if isinstance(call, dict) else call.function
        args_raw = func['arguments'] if isinstance(func, dict) else func.arguments
        args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
        keyword = args.get('artwork_keyword', '')
    except Exception:
        return "I am sorry, but I could not understand your request"

    artwork = get_artwork(keyword)
    try:
        parsed = json.loads(artwork)
        titles = [item.get('title') for item in parsed.get('data', []) if item.get('title')]
        if not titles:
            return "No recommendations found."
        return f"Here are some recommendations: {titles}"
    except Exception:
        return "Apologies, I could not make any recommendation based on the request."
        

user_message = """I don't have much time to visit the museum and would like some recommendations.
I like the seaside and quiet places"""

print("RESPONSE: ", get_response(user_message))
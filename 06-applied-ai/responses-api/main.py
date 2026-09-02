import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def response(input: str):
    return client.responses.create(
        model="gpt-5.4-mini",
        reasoning={
            "effort": "none",   # options = none | low | medium | high | xhigh
            "summary": "auto"
        }, 
        max_output_tokens=300,
        instructions="Be clear and concise",
        input=input
    )

def main(query):
    resp = response(query)
    output = resp.output_text
    tokens = resp.usage.output_tokens
    id = resp.id
    extract_all_items(resp)
    
    
def extract_all_items(resp):
    items = [item for item in resp.output]
    print(items)
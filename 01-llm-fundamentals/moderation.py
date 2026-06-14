import os
import uuid
import asyncio
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

unique_id = str(uuid.uuid4())

def is_request_safe(prompt: str) -> bool:
    mod_resp =  client.moderations.create(input=prompt)
    return mod_resp.results[0].categories.violence

def llm_response(messages: str):
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        user=unique_id
    )
    return resp.choices[0].message.content

async def generate_response(query: str, messages: str):
    resp = (llm_response(messages) if is_request_safe(query) 
        else "Sorry, I can not answer this question. Please, let's talk about something else")
    return str(resp)
    
    
query = """
...until someone throws an exploding Kitten.
When that happens, that person explodes. They are now dead.
This process continues until...
"""

guardrail = """
Your role is to assess whether the user question is allowed or not.
The allowed topics are games only. if the ropic is allowed, reply as normal, otherwise say
'Apologies, but the topic is not allowed.' """
   
    
messages = [
    {"role": "system", "content": guardrail},
    {"role": "user", "content": query}
]

async def main():
    safe = is_request_safe(query)
    response = await generate_response(query, messages)
    print("Query safe: ", safe)
    print("Response: ", response)



if __name__ == "__main__":
    asyncio.run(main())
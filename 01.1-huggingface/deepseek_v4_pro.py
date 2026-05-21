import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(
    api_key=os.getenv("HF_TOKEN")
)

def get_response(prompt):
    response = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V4-Pro:novita",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],    
    )
    return response.choices[0].message


query = "What is the capital of France?"
response = get_response(query)
print(response)
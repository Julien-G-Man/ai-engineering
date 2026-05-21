import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

client = InferenceClient(
    provider="together", # together.ai
    token=os.getenv("HF_TOKEN")
)

def get_completion(query):
    completion = client.chat.completions.create(
        model="deepseek-ai/DeepSeek-V3",
        messages=[
            {"role": "user", "content": query}
        ],
    )
    return completion.choices[0].message

query = "What's the capital of France?"
result = get_completion(query)
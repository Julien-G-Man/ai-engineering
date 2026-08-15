import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def main(prompt, image_url, model="gpt-4o-mini"):
    response = client.responses.create(
        model=model,
        input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_url},
                ],
            }],
    )
    return response.output_text


image_url="https://assets.bucketlistly.blog/sites/5adf778b6eabcc00190b75b1/content_entry5b155bed5711a8176e9f9783/66a6484ea4f94d0002788638/files/peru-travel-photo-20240727203150561-main-image.jpg"
prompt = "What is on this image?"

if __name__ == "__main__":
    print(main(prompt, image_url))
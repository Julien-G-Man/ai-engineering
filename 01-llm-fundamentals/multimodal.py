import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model="gpt-3.5-turbo"

def main(prompt, image_ur, model):
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"{image_ur}"},
                ],
            }
        ],  
    )
    return response.choices[0].message.content


image_url="https://instagram.facc7-1.fna.fbcdn.net/v/t51.82787-15/590377644_18063940265551395_6043744558244201288_n.jpg?stp=dst-jpg_e35_p640x640_sh0.08_tt6&_nc_cat=108&ig_cache_key=Mzc4Mzg3OTMyNTUzNjU1MzA2NA%3D%3D.3-ccb7-5&ccb=7-5&_nc_sid=58cdad&efg=eyJ2ZW5jb2RlX3RhZyI6InhwaWRzLjE0NDB4MTkyMC5zZHIuQzMifQ%3D%3D&_nc_ohc=1rhjSSQDYJ8Q7kNvwF6GpBL&_nc_oc=AdoUbuobg9j8wyavjW1nkNMT4Hvssp4WudPbX7Z14I0FcgMOJR1jN4ONJdQjavZ2DZc&_nc_ad=z-m&_nc_cid=0&_nc_zt=23&_nc_ht=instagram.facc7-1.fna&_nc_gid=LvR1DzVQDw_4OX5z0kDJpg&_nc_ss=7a22e&oh=00_Af2FrucEeXLqTmCjMQhcqJCDvLUpCcIxOwxrrGkD_qnVqw&oe=69F25792"
prompt = "What is on this image?"

if __name__ == "__main__":
    main(prompt, image_url, model)
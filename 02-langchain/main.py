import os
import datetime
from dotenv import load_dotenv
from openai import OpenAI

current_date = datetime.datetime.now().date()

# Define the date after which the model should be set to "gpt-3.5-turbo"
target_date = datetime.date(2024, 6, 12)

if current_date > target_date:
    llm_model = "gpt-3.5-turbo"
else: 
    llm_model = "gpt-3.5-turbo-0301"

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_completion(prompt, model=llm_model):
    messages=[
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.2
    )
    return response.choices[0].messages.content


customer_email = """
Arrr, I be fuming that me blender lid \
flew off and splattered me kitchen walls \
with smoothie! And to make matters worse,\
the warranty don't cover the cost of \
cleaning up me kitchen. I need yer help \
right now, matey!
"""

style = """American English \
in a calm and respectful tone
"""

prompt = f"""Translate the text \
that is delimited by triple backticks 
into a style that is {style}.
text: ```{customer_email}```
"""

print(prompt)

from langchain_openai import ChatOpenAI

chat = ChatOpenAI(temperature=0.0, model=llm_model)

template_string = """Translate the text \
that is delimited by triple backticks \
into a style that is {style}. \
text: ```{text}```
"""

from langchain_core.prompts import ChatPromptTemplate
prompt_template = ChatPromptTemplate.from_template(template_string)
prompt_template.messages[0].prompt
prompt_template.messages[0].prompt.input_variables

customer_style = """American English \
in a calm and respectful tone
"""
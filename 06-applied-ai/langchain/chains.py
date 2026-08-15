import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import pandas as pd
df = None
try:
    file = Path(__file__).resolve().parents[1] / "data" / "data.csv"
    df = pd.read_csv(file)
    df.head()
except Exception as e:
    print(e)

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from llm_factory import get_llm

product = "Queen Size Sheet Set"


def llm_chain():
    llm = get_llm(temperature=0.9)
    prompt = ChatPromptTemplate.from_template(
        "What is the best name to describe a company that makes {product}"
    )
    chain = prompt | llm | StrOutputParser()
    print(chain.invoke({"product": product}))

# modern - using pipe operator instead of langchain_classic.chains
# chain = prompt | model
# response = chain.invoke({"topic": "space"})


def simple_sequential_chain():
    llm = get_llm(temperature=0.9)

    prompt_1 = ChatPromptTemplate.from_template(
        "What is the best name to describe a company that makes {product}"
    )
    prompt_2 = ChatPromptTemplate.from_template(
        "Write a 20 words description for the following \
        company: {company_name}"
    )

    chain_one = prompt_1 | llm | StrOutputParser()
    chain_two = prompt_2 | llm | StrOutputParser()

    company_name = chain_one.invoke({"product": product})
    description = chain_two.invoke({"company_name": company_name})

    print("Company name:", company_name)
    print("Description:", description)


def regular_sequential_chain():
    llm = get_llm(temperature=0.9)
    if df is None:
        raise RuntimeError("CSV data is unavailable. Put data.csv in AI-Engineering/data/.")

    prompt_1 = ChatPromptTemplate.from_template(
        "Translate the following review to english:"
        "\n\n{Review}"
    )

    prompt_2 = ChatPromptTemplate.from_template(
        "Can you summarize the following review in 1 sentence:"
        "\n\n{English_Review}"
    )

    prompt_3 = ChatPromptTemplate.from_template(
        "What language is the following review:\n\n{Review}"
    )

    prompt_4 = ChatPromptTemplate.from_template(
        "Write a follow up response to the following "
        "summary in the specified language:"
        "\n\nSummary: {summary}\n\nLanguage: {language}"
    )

    chain_one = prompt_1 | llm | StrOutputParser()
    chain_two = prompt_2 | llm | StrOutputParser()
    chain_three = prompt_3 | llm | StrOutputParser()
    chain_four = prompt_4 | llm | StrOutputParser()

    review = df["Review"].iloc[5] if len(df) > 5 else df["Review"].iloc[0]

    english_review = chain_one.invoke({"Review": review})
    summary = chain_two.invoke({"English_Review": english_review})
    language = chain_three.invoke({"Review": review})
    followup_message = chain_four.invoke({"summary": summary, "language": language})

    print("English review:", english_review)
    print("Summary:", summary)
    print("Language:", language)
    print("Followup:", followup_message)
    

    
MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
language model select the model prompt best suited for the input. \
You will be given the names of the available prompts and a \
description of what the prompt is best suited for. \
You may also revise the original input if you think that revising\
it will ultimately lead to a better response from the language model.

<< FORMATTING >>
Return a markdown code snippet with a JSON object formatted to look like:
```json
{{{{
    "destination": string \\ "DEFAULT" or name of the prompt to use in {destinations}
    "next_inputs": string \\ a potentially modified version of the original input
}}}}
```

REMEMBER: The value of “destination” MUST match one of \
the candidate prompts listed below.\
If “destination” does not fit any of the specified prompts, set it to “DEFAULT.”
REMEMBER: "next_inputs" can just be the original input \
if you don't think any modifications are needed.

<< CANDIDATE PROMPTS >>
{destinations}

<< INPUT >>
{{input}}

<< OUTPUT (remember to include the ```json)>>"""
    
physics_template = """You are a very smart physics professor. \
You are great at answering questions about physics in a concise\
and easy to understand manner. \
When you don't know the answer to a question you admit\
that you don't know.

Here is a question:
{input}"""
    
math_template = """You are a very good mathematician. \
You are great at answering math questions. \
You are so good because you are able to break down \
hard problems into their component parts,
answer the component parts, and then put them together\
to answer the broader question.

Here is a question:
{input}"""

history_template = """You are a very good historian. \
You have an excellent knowledge of and understanding of people,\
events and contexts from a range of historical periods. \
You have the ability to think, reflect, debate, discuss and \
evaluate the past. You have a respect for historical evidence\
and the ability to make use of it to support your explanations \
and judgements.

Here is a question:
{input}"""
    
computerscience_template = """You are a successful computer scientist.\
You have a passion for creativity, collaboration,\
forward-thinking, confidence, strong problem-solving capabilities,\
understanding of theories and algorithms, and excellent communication \
skills. You are great at answering coding questions. \
You are so good because you know how to solve a problem by \
describing the solution in imperative steps \
that a machine can easily interpret and you know how to \
choose a solution that has a good balance between \
time complexity and space complexity.

Here is a question:
{input}"""
     
      
def router_chain(query):    
    llm = get_llm(temperature=0.9)
    prompt_infos = [
        {
            "name": "physics", 
            "description": "Good for answering questions about physics", 
            "prompt_template": physics_template
        },
        {
            "name": "math", 
            "description": "Good for answering math questions", 
            "prompt_template": math_template
        },
        {
            "name": "History", 
            "description": "Good for answering history questions", 
            "prompt_template": history_template
        },
        {
            "name": "computer science", 
            "description": "Good for answering computer science questions", 
            "prompt_template": computerscience_template
        }
    ]
    
    destination_chains = {}

    for p_info in prompt_infos:
        name = p_info.get("name")
        prompt_template = p_info.get("prompt_template")
        prompt = ChatPromptTemplate.from_template(template=prompt_template)
        chain = prompt | llm | StrOutputParser()
        destination_chains[name.lower()] = chain

    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = "\n".join(destinations)

    default_prompt = ChatPromptTemplate.from_template("{input}")
    default_chain = default_prompt | llm | StrOutputParser()

    class RouteDecision(BaseModel):
        destination: str = Field(description="Prompt name from candidates, or DEFAULT")
        next_inputs: str = Field(description="Original or lightly rewritten input")

    router_prompt = ChatPromptTemplate.from_template(
        """Route the user input to the best destination.

Valid destinations:
{destinations}

Rules:
- Return destination exactly as one of the names above, or DEFAULT.
- Keep next_inputs as the original input unless a light rewrite helps.

User input:
{input}
"""
    )

    router_chain = router_prompt | llm.with_structured_output(RouteDecision)
    route = router_chain.invoke(
        {"destinations": destinations_str, "input": query}
    )

    destination_key = (route.destination or "DEFAULT").strip().lower()
    routed_input = (route.next_inputs or query).strip()
    selected_chain = destination_chains.get(destination_key)

    if selected_chain is None:
        print("default:", {"input": routed_input})
        result = default_chain.invoke({"input": routed_input})
    else:
        print(f"{destination_key}:", {"input": routed_input})
        result = selected_chain.invoke({"input": routed_input})

    print(result)
    

if __name__ == "__main__":
    try:
        router_chain("What is black body radiation?")
        router_chain("Why do cells contain DNA?")
    except RuntimeError as e:
        print(e)
    


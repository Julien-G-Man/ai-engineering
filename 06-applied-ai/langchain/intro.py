from langchain_core.prompts import ChatPromptTemplate
from llm_factory import get_llm

customer_email = """
Arrr, I be fuming that me blender lid
flew off and splattered me kitchen walls
with smoothie! And to make matters worse,
the warranty don't cover the cost of
cleaning up me kitchen. I need yer help
right now, matey!
"""

customer_style = "American English in a calm and respectful tone"

service_reply = """
Hey there customer,
the warranty does not cover cleaning expenses for your kitchen
because it's your fault that you misused your blender
by forgetting to put the lid on before starting the blender.
Tough luck! See ya!
"""

service_style_pirate = "a polite tone that speaks in English Pirate"

# PROMPT: use placeholders so you can reuse the same prompt for many inputs.
template_string = """
Translate the text between triple backticks into a style that is {style}.
text: ```{text}```
"""

prompt_template = ChatPromptTemplate.from_template(template_string)


def print_prompt_concept() -> None:
    print("=== Prompt Concept ===")
    print("Template variables:", prompt_template.messages[0].prompt.input_variables)
    print()


def run_demo() -> None:
    print_prompt_concept()

    try:
        llm = get_llm(temperature=0.0)
    except RuntimeError as e:
        print(e)
        print("This file still demonstrates how prompts are structured.")
        return

    customer_messages = prompt_template.format_messages(
        style=customer_style,
        text=customer_email,
    )
    
    customer_response = llm.invoke(customer_messages)
    print("=== Customer Rewrite ===")
    print(customer_response.content)
    print()

    service_messages = prompt_template.format_messages(
        style=service_style_pirate,
        text=service_reply,
    )
    
    service_response = llm.invoke(service_messages)
    print("=== Service Reply Rewrite ===")
    print(service_response.content)


if __name__ == "__main__":
    run_demo()
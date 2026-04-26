""" Models, Prompts and Parsers """

import os
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from llm_factory import get_llm


class ReviewAnalysis(BaseModel):
    gift: bool = Field(description="Was this purchased as a gift?")
    delivery_days: int = Field(description="Days to delivery, or -1 if unknown")
    price_value: list[str] = Field(description="Sentences discussing price or value")


parser = PydanticOutputParser(pydantic_object=ReviewAnalysis)

customer_review = """
This leaf blower is pretty amazing. It has four settings:
candle blower, gentle breeze, windy city, and tornado.
It arrived in two days, just in time for my wife's anniversary present.
I think my wife liked it so much she was speechless.
So far I've been the only one using it, and I've been
using it every other morning to clear the leaves on our lawn.
It's slightly more expensive than the other leaf blowers
out there, but I think it's worth it for the extra features.
"""

review_template = """
Extract the requested fields from the text.

Text:
{text}

{format_instructions}
"""

prompt = ChatPromptTemplate.from_template(review_template)


def run_demo() -> None:
    print("=== Parser Format Instructions ===")
    print(parser.get_format_instructions())
    print()

    try:
        llm = get_llm(temperature=0.0)
    except RuntimeError as e:
        print(e)
        print("Parser setup is valid; only the live API call is skipped.")
        return

    # CHAIN: prompt -> model -> parser returns a typed Pydantic object.
    chain = prompt | llm | parser
    parsed: ReviewAnalysis = chain.invoke(
        {
            "text": customer_review,
            "format_instructions": parser.get_format_instructions(),
        }
    )

    print("=== Parsed Output (Typed) ===")
    print(parsed.model_dump())
    print()
    print("gift:", parsed.gift)
    print("delivery_days:", parsed.delivery_days)
    print("price_value:", parsed.price_value)


if __name__ == "__main__":
    run_demo()




import os, json
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

sys_prompt = "You are a helpful Teacher who provides concise, personalized explanations. Be very brief."


tools = [
    {
        "type": "function",
        "name": "convert_currency",
        "description": "Convert an amount form one currency to another using real-time exchange rates.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount":        {"type": "number", "description": "The amount of money to convert"},
                "from_currency": {"type": "string", "description": "The source currency code (e.g, 'USD, 'EUR)"},
                "to_currency":   {"type": "string", "description": "The target currency code (e.g., 'USD', 'EUR')"}
            },
            "required": ["amount", "from_currency", "to_currency"],
            "additionalProperties": False
        }
    },
]


def convert_currency(amount, from_currency, to_currency):
    url = f"https://api.frankfurter.dev/v1/latest?base={from_currency}&symbols={to_currency}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        rate = data["rates"].get(to_currency)
        
        if rate is None:
            return f"Could not find exchange rate for {from_currency} to {to_currency}"
        
        converted_amount = amount * rate
        return f"{amount} {from_currency} = {converted_amount:.2f} {to_currency} (Rate: {rate})"
    except requests.exceptions.RequestException as e:
        return f"Error converting currency: {str(e)}"


def response(messages):
    return client.responses.create(
        model="gpt-5.4-mini",
        tools=tools,        
        reasoning={
            "effort": "none",   # options = none | low | medium | high | xhigh
            "summary": "auto"
        }, 
        max_output_tokens=300,
        instructions=sys_prompt,
        input=messages
    )


def main():
    messages = [{"role": "system", "content": sys_prompt}]
    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() == "exit":
                break
            
            messages.append({"role": "user", "content": user_input})
            resp = response(messages)
            messages += resp.output
            
            tool_called = False
            
            for item in resp.output:
                if item.type == "function_call":
                    tool_called = True
                    if item.name == "convert_currency":
                        currency_result = convert_currency(**json.loads(item.arguments))
                        
                        messages.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": json.dumps({"convert_currency": currency_result})
                        })
                        
                        final_resp = response(messages)
                        messages += final_resp.output
                        
                        print(f"\nAssistant: {final_resp.output_text}\n")
                        break
            if not tool_called:
                print(f"\nAssistant: {resp.output_text}\n")
    except Exception as e:
        print(f"Error: {str(e)}")



if __name__ == "__main__":
    main()
import os
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

function_definition = [{
    "type": "function",
    "function": {
        "name": "extract_job_info",
        "description": "Get the job information from the body of the input text",
        "parameters": {
            "type": "object",
            "properties": {
                "job": { "type": "string", "description": "Job title"},
                "location": { "type": "string", "description": "Office location"},
            }
        }
    }
}]

time_zone_function = {
    'type': 'function',
    'function': {
        'name': 'get_timezone',
        'description': 'Return the timezone corresponding to the location in the job posting',
        'parameters': {
            'type': 'object',
            'properties': {
                'timezone': {
                    'type': 'string', 
                    'description': 'Timezone'}
            }
        }
    }
}

function_definition.append(time_zone_function)

def get_response_with_function_call(user_prompt: str, tool_num: int):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
                You are a concise extractor that pulls structured job information from freeform text.
                Return only the requested fields when calling the function: `job` and `location`.
                Respond with a JSON object when returning the results.
                If a field is not present, return an empty string for it."""
            },
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        tools=function_definition,
        tool_choice={'type': 'function', 
                     'function': {'name': 'extract_job_info'}}
    )

    # Safely obtain tool calls list
    try:
        tool_calls = response.choices[0].message.tool_calls
    except Exception:
        tool_calls = []

    if not tool_calls:
        # nothing to parse — return normalized empty structure
        return {"job": "", "location": "", "timezone": ""}

    # Convert user-provided 1-based index into a safe 0-based index.
    requested_index = (tool_num - 1) if isinstance(tool_num, int) else None
    if requested_index is None or not (0 <= requested_index < len(tool_calls)):
        # fallback to the last call if requested index is invalid
        selected_call = tool_calls[-1]
    else:
        selected_call = tool_calls[requested_index]

    raw_args = selected_call.function.arguments
    try:
        parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    except Exception:
        # if parsing fails, return the raw string under a key for inspection
        return {"job": "", "location": "", "timezone": "", "raw": raw_args}

    # Normalize output keys and ensure presence
    return {
        "job": parsed.get("job", ""),
        "location": parsed.get("location", ""),
        "timezone": parsed.get("timezone", ""),
    }



query = (
            "Extract the job title and office location from the following posting:\n\n"
            "\"We are hiring a Senior Data Engineer to join our analytics team in London. "
            "This remote-friendly role requires 5+ years in data pipelines and strong SQL skills.\""
        )

first_response = get_response_with_function_call(query, 1)
second_response = get_response_with_function_call(query, 2)

if __name__ == "__main__":
    print("1st RESPONSE:", repr(first_response))
    print("2nd RESPONSE:", repr(second_response))
    print(len(function_definition))

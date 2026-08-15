from transformers import pipeline

def get_results(query):
    gpt2_pipeline = pipeline(task="text-generation", model="openai-community/gpt2")
    return gpt2_pipeline(query, max_tokens=10, num_return_sequences=2)



if __name__ == "__main__":
    query = "What is AI?"
    results = get_results(query)
    for result in results:
        print(result.get("generated-text", "none"))   
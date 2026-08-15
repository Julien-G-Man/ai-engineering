import tiktoken

model="gpt-4o-mini"
encoding = tiktoken.encoding_for_model(model)
prompt = """Tokens can be full words, or groups of characters 
commonly grouped together: tokenization."""

num_tokens = len(encoding.encode(prompt))
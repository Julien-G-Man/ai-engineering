"""
Memory types highlights:
1. ConversationBufferMemory
    Stores the full conversation history in order.
    Best when chats are short and you want maximum context.

2. ConversationBufferWindowMemory
    Keeps only the last k interaction pairs.
    Useful to reduce prompt size while preserving recent context.

3. ConversationTokenBufferMemory
    Keeps conversation history under a token limit.
    Good when context budget matters and message lengths vary.

4. ConversationSummaryMemory
    Summarizes older chat history into a compact running summary.
    Useful for long conversations where full history is too expensive.

Additional memory categories:
- Vector data memory: retrieves relevant past information via embeddings.
- Entity memory: tracks facts about specific people, places, or things.

Notes:
- You can combine multiple memory strategies in one app (for example, short-term window memory plus long-term vector memory).
- You can also persist memory in a conventional SQL database (for example, PostgreSQL/MySQL/SQLite)
"""

import os
from langchain_classic.chains.conversation.base import ConversationChain
from langchain_classic.memory.buffer import ConversationBufferMemory
from llm_factory import get_llm
    
def run_demo() -> None:
    try:
        llm = get_llm(temperature=0.0)
    except RuntimeError as e:
        print(e)
        return

    memory = ConversationBufferMemory()
    conversation = ConversationChain(
        llm=llm, 
        memory=memory, 
        verbose=True
    )

    print("=== Turn 1 ===")
    print(conversation.predict(input="Hi, my name is Ada."))
    print("=== Turn 2 ===")
    print(conversation.predict(input="What is my name?"))
    print(memory.buffer)
    memory.save_context(conversation)
    memory.load_memory_variables({})


# explicitly

# helper funtion I made instead of having to save and 
# load over and over, not necessary :)
def save_and_load(memory: object, input: str, output: str):
    memory.save_context({"input": input}, {"output": output})
    memory.load_memory_variables({})


def explicit():
    llm = get_llm(temperature=0.0)
    memory = ConversationBufferMemory()
    conversation = ConversationChain(
        llm=llm,
        memory=memory,
        verbose=False
    )
    save_and_load(memory, "Hi", "What's up")
    print(memory.buffer)
    
    save_and_load(memory, "Nothing much", "Cool")
    print(memory.buffer)
    

def with_buffer_window():
    llm = get_llm(temperature=0.0)
    from langchain_classic.memory.buffer_window import ConversationBufferWindowMemory
    memory = ConversationBufferWindowMemory(k=2)
    save_and_load(memory, "Hi", "What's up")


def with_token_buffer_memory():
    llm = get_llm(temperature=0.0)
    from langchain_classic.memory import ConversationTokenBufferMemory
    memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=50)
    memory.save_context({"input": "AI is what?"}, {"ouput": "Amazing!"})
    memory.save_context({"input": "Backpropagation is what?"}, {"output": "Beautiful"})
    memory.save_context({"input": "Chatbot are what?"}, {"output": "Charming"})
    memory.load_memory_variables({})

    
    
# create a long string
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."


def with_summary_memory():
    llm = get_llm(temperature=0.0)
    # keeps explicit chat memory <= set token limit
    from langchain_classic.memory import ConversationSummaryMemory
    memory = ConversationSummaryMemory(llm=llm, max_token_limit=100)
    memory.save_context({"input": "Hello"}, {"output": "What's up"})
    memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
    memory.save_context({"input": "What is on the schedule today?"}, 
                    {"output": f"{schedule}"})
    memory.load_memory_variables({})



if __name__ == "__main__":
    with_summary_memory()

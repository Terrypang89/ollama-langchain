import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage

# Load environment variables from .env
load_dotenv()
ollama_api_base = os.getenv("OLLAMA_API_BASE")
llm = ChatOllama(
    model="mistral:7b",
    temperature=0,
    base_url=ollama_api_base
)

messages = [
    (
        "system",
        "You are expert in identifying gibberish words in a sentence. If a sentence contains gibberish return response as 'Not eligible'",
    ),
    ("human", "good morning"),
]
ai_msg = llm.invoke(messages)

print("\nresult \n",ai_msg)

print("\nresult ",ai_msg.content)
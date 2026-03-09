import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings

load_dotenv()
OLLAMA_SERVER = os.getenv("OLLAMA_API_BASE")

# EA_LOG_PATH = r"C:\Users\Tofy3\AppData\Roaming\MetaQuotes\Tester\02FA8E9A84D4D4A59CB903141393B86D\Agent-127.0.0.1-3000\logs"
EA_LOG_PATH = os.getenv(r"EA_LOG_PATH")

# Pick the latest log file
log_dir = Path(EA_LOG_PATH)
latest_log = max(log_dir.glob("*.log"), key=lambda f: f.stat().st_mtime)

with open(latest_log, "r", encoding="utf-16") as f:
    log_text = f.read()

# Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_text(log_text)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=f"{OLLAMA_SERVER}"
)

vectorstore = FAISS.from_texts(chunks, embeddings)
vectorstore.save_local("logs_index")
print(f"Indexed {len(chunks)} chunks from {latest_log.name}")
print("Vectorstore size:", vectorstore.index.ntotal)

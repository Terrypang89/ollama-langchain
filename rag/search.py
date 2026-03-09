import os
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import re
import chardet

def read_file_auto(path):
    # Read raw bytes
    with open(path, "rb") as f:
        raw = f.read()
    # Detect encoding
    enc = chardet.detect(raw)["encoding"]
    if enc is None:
        enc = "utf-8"  # fallback
    # Decode safely
    return raw.decode(enc, errors="ignore")

def load_and_chunk_mq5(main_file, header_files, header_path):
    # Load headers
    headers_code = ""
    for header in header_files:
        full_path = os.path.join(header_path, header)
        headers_code += f"\n// HEADER: {header}\n" + read_file_auto(full_path)

    # Load main EA
    mq5_code = read_file_auto(main_file)

    # Combine
    full_code = headers_code + "\n\n// MAIN EA FILE\n" + mq5_code

    # Split into chunks by function definitions (more general regex)
    chunks = re.split(r"([a-zA-Z_][a-zA-Z0-9_]*\s+\w+\(.*?\)\s*{)", full_code)

    # Reattach function headers to their bodies
    structured_chunks = [chunks[0]]  # keep pre-function code
    for i in range(1, len(chunks)-1, 2):
        structured_chunks.append(chunks[i] + chunks[i+1])

    return structured_chunks


load_dotenv()

OLLAMA_SERVER = os.getenv("OLLAMA_API_BASE")
EA_QML_FILE = os.getenv("EA_QML_FILE")
EA_HEADER_PATH = os.getenv("EA_HEADER_PATH")

header_files = ["TofyTrade.mqh", "TofyInclude.mqh"]

# Models
mistral = OllamaLLM(model="mistral:7b", base_url=OLLAMA_SERVER)
claude_code = OllamaLLM(model="claude-code", base_url=OLLAMA_SERVER)

# Embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_SERVER)

# Load vector store (with explicit deserialization flag)
vectorstore = FAISS.load_local(
    "logs_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Prompt for retrieval QA
prompt = PromptTemplate.from_template(
    "Use the following context to answer the question:\n{context}\n\nQuestion: {question}\nAnswer:"
)

# Runnable pipeline
retriever = vectorstore.as_retriever()
qa_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | mistral
    | StrOutputParser()
)

# Step 1: Analyze logs
analysis = qa_chain.invoke("Summarize weaknesses in this trading strategy")

mq5_code = load_and_chunk_mq5(EA_QML_FILE, header_files, EA_HEADER_PATH)

refined_code = claude_code.invoke(
    f"Refine this MQL5 EA based on analysis:\n{analysis}\nOriginal code:\n{mq5_code}"
)

print("Analysis:", analysis)
print("Refined EA Code:", refined_code)


prompt = PromptTemplate.from_template(
    "You are an expert in MQL5 trading systems. Based on the following weaknesses:\n{analysis}\n\nRefactor the EA code below to improve risk management, trade filtering, and readability:\n{mq5_code}"
)

refined_code = claude_code.invoke(prompt.format(analysis=analysis, mq5_code=mq5_code))

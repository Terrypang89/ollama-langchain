import subprocess
import Path
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings, OllamaLLM

def generate_version_id(vectorstore):
    # Search for all logs (or use metadata query)
    docs = vectorstore.similarity_search("latest backtest", k=10)
    # Extract existing version_ids
    version_ids = [int(d.metadata.get("version_id", 0)) for d in docs if "version_id" in d.metadata]
    return max(version_ids, default=0) + 1

def store_log(vectorstore, metrics, version_id):
    # Find latest log file
    log_dir = Path(EA_LOG_PATH)
    latest_log = max(log_dir.glob("*.log"), key=lambda f: f.stat().st_mtime)

    # Read log file (UTF-16 encoding typical for MT5 logs)
    with open(latest_log, "r", encoding="utf-16") as f:
        log_text = f.read()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(log_text)

    # Create embeddings
    embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=f"{OLLAMA_SERVER}")

    # Build vector store from chunks
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # Save locally for persistence
    vectorstore.save_local("logs_index")

    # Add metadata for version tracking
    vectorstore.add_texts(
        texts=[log_text],
        metadatas=[{
            "version_id": version_id,
            "timestamp": datetime.now().isoformat(),
            "profit_factor": metrics["profit_factor"],
            "drawdown": metrics["drawdown"],
            "win_rate": metrics["win_rate"],
            "type": "log"
        }],
        ids=[f"log_{version_id}"]
    )

    return vectorstore

# --- Retrieve Last Two Logs ---
def get_last_two_logs(vectorstore):
    docs = vectorstore.similarity_search("latest backtest", k=2)
    if len(docs) < 2:
        # Handle cases where fewer than 2 logs exist
        print("Not enough logs in vector store to compare.")
        return None, None

    # Return latest and previous
    return docs[0], docs[1]


# --- Compare Logs with Ollama ---
def compare_logs(ollama_model, latest_log, previous_log):
    
    if latest_log != None and previous_log != None:
        mistral = OllamaLLM(model="mistral:7b", base_url=OLLAMA_SERVER)
        prompt = f"""
    Compare these two backtests:

    Latest:
    PF={latest_log.metadata['profit_factor']}, DD={latest_log.metadata['drawdown']}, WR={latest_log.metadata['win_rate']}
    Log: {latest_log.page_content}

    Previous:
    PF={previous_log.metadata['profit_factor']}, DD={previous_log.metadata['drawdown']}, WR={previous_log.metadata['win_rate']}
    Log: {previous_log.page_content}

    Has performance improved? Summarize differences and give a clear verdict.
    """
        return mistral.invoke(prompt)
    else:
        return None
    
def analyze_losses_and_skips(ollama_model, latest_log, version_id):
    prompt = f"""
    Examine this backtest log (version {version_id}):

    {latest_log.page_content}

    Tasks:
    - Identify trades that ended in losses
    - Identify signals not executed
    - Summarize weak areas
    """
    analysis = ollama_model.invoke(prompt)
    return {"version_id": version_id, "analysis": analysis}

def refine_code_with_analysis(ollama_model, latest_log, ea_code, user_prompt, previous_log, version_id):
    analysis = analyze_losses_and_skips(ollama_model, latest_log, version_id)

    prompt = f"""
    Version {version_id} refinement:
    Based on this analysis:
    {analysis['analysis']}

    Refine the following EA code:
    {ea_code}

    User guidance:
    {user_prompt}
    """
    refined_code = ollama_model.invoke(prompt)

    verdict = None
    if previous_log:
        verdict = compare_logs(ollama_model, latest_log, previous_log)

    return {
        "version_id": version_id,
        "refined_code": refined_code,
        "analysis": analysis,
        "verdict": verdict
    }

def store_refined_code(vectorstore, refined_code, version_id):
    vectorstore.add_texts(
        texts=[refined_code],
        metadatas=[{"version_id": version_id, "type": "refined_code"}],
        ids=[f"code_{version_id}"]
    )

def store_verdict(vectorstore, verdict_text, version_id):
    vectorstore.add_texts(
        texts=[verdict_text],
        metadatas=[{"version_id": version_id, "type": "verdict"}],
        ids=[f"verdict_{version_id}"]
    )

# --- Committer Agent ---
def commit_changes(files, message="Auto-refined EA + updated .set + log"):
    subprocess.run(["git", "add"] + files)
    subprocess.run(["git", "commit", "-m", message])
    subprocess.run(["git", "push", "origin", "main"])

def run_pipeline(vectorstore, ollama_model, ea_file, header_files, set_file, log_text, metrics, ea_code, user_prompt=None):
    # Step 1: Generate new version ID
    version_id = generate_version_id(vectorstore)

    # Step 2: Store log
    store_log(vectorstore, log_text, metrics, version_id)

    # Step 3: Retrieve last two logs
    latest_log, previous_log = get_last_two_logs(vectorstore)

    # Step 4: Refine code with analysis + user guidance
    refined_code, verdict = refine_code_with_analysis(
        ollama_model, latest_log, ea_code, user_prompt, previous_log, version_id
    )

    # Step 5: Store analysis, refined code, and verdict
    store_analysis(vectorstore, refined_code, version_id)  # analysis already inside refine_code
    store_refined_code(vectorstore, refined_code, version_id)
    if verdict:
        store_verdict(vectorstore, verdict, version_id)

    print(f"--- Version {version_id} ---")
    print("Refined EA Code:\n", refined_code)
    if verdict:
        print("Comparison Verdict:\n", verdict)
        if "improved" in verdict.lower():
            commit_changes([ea_file] + header_files + [set_file], f"Commit version {version_id} after improvement")
        else:
            print("No improvement detected. Commit skipped.")
    else:
        print("Baseline run — no previous log to compare yet.")

def get_artifacts_by_version(vectorstore, version_id):
    # Search for all items tagged with this version_id
    docs = vectorstore.similarity_search(f"version {version_id}", k=10)

    # Organize by type
    artifacts = {"log": None, "analysis": None, "refined_code": None, "verdict": None}
    for d in docs:
        t = d.metadata.get("type")
        if t == "analysis":
            artifacts["analysis"] = d.page_content
        elif t == "refined_code":
            artifacts["refined_code"] = d.page_content
        elif t == "verdict":
            artifacts["verdict"] = d.page_content
        else:
            # default to log if no type set
            artifacts["log"] = d.page_content

    return artifacts

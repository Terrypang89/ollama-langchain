import subprocess
import Path
import os
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
from langchain.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
# from langchain.embeddings import OllamaEmbeddings

load_dotenv()
OLLAMA_SERVER = os.getenv("OLLAMA_API_BASE")

def generate_version_id(vectorstore):
    # Search for all logs (or use metadata query)
    docs = vectorstore.similarity_search("latest backtest", k=10)
    # Extract existing version_ids
    version_ids = [int(d.metadata.get("version_id", 0)) for d in docs if "version_id" in d.metadata]
    return max(version_ids, default=0) + 1

def store_log(vectorstore, log_text, metrics, version_id):

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

def store_analysis(vectorstore, analysis_text, version_id):
    """
    Store the analysis output into the vector store with metadata.
    """
    vectorstore.add_texts(
        texts=[analysis_text],
        metadatas=[{
            "version_id": version_id,
            "type": "analysis",
            "timestamp": datetime.now().isoformat()
        }],
        ids=[f"analysis_{version_id}"]
    )

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
        ollama_model = OllamaLLM(model="mistral:7b", base_url=OLLAMA_SERVER)
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
        return ollama_model.invoke(prompt)
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

def parse_response_sections(response_text):
    """
    Split AI response into analysis, code, and verdict sections.
    Assumes the model outputs markers like:
    ### Analysis
    ### Refined Code
    ### Verdict
    """
    analysis, code, verdict = "", "", ""
    sections = response_text.split("###")
    for section in sections:
        if section.lower().startswith("analysis"):
            analysis = section.replace("Analysis", "").strip()
        elif section.lower().startswith("refined code"):
            code = section.replace("Refined Code", "").strip()
        elif section.lower().startswith("verdict"):
            verdict = section.replace("Verdict", "").strip()
    return analysis, code, verdict

def refine_code_with_analysis(
    ollama_model,
    latest_log,
    ea_code,
    user_prompt,
    previous_log,
    version_id
):
    """
    Refine EA code using analysis of logs and optional user guidance.
    Returns: (refined_code, verdict, analysis_text)
    """

    # --- Build prompt ---
    prompt = f"""
    You are an expert in trading strategy refinement.
    Task: Analyze the latest EA log and refine the EA code.

    Version: {version_id}

    Latest Log:
    {latest_log}

    Previous Log:
    {previous_log if previous_log else "None"}

    Current EA Code:
    {ea_code}

    User Guidance:
    {user_prompt if user_prompt else "None"}

    Please provide:
    1. Analysis of weaknesses and skipped trades.
    2. Suggested refinements to the EA code.
    3. A verdict comparing this version to the previous one (improved, worse, or baseline).
    """

    # --- Call Ollama with chosen model ---
    llm = Ollama(model=ollama_model, base_url="http://localhost:11434")
    response = llm(prompt)

    # --- Parse response ---
    # You can design a simple parser if you format the AI output consistently
    analysis_text, refined_code, verdict = parse_response_sections(response)

    return refined_code, verdict, analysis_text

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

def store_artifact(vectorstore, text, version_id, artifact_type):
    vectorstore.add_texts(
        texts=[text],
        metadatas=[{
            "version_id": version_id,
            "type": artifact_type,
            "timestamp": datetime.now().isoformat()
        }],
        ids=[f"{artifact_type}_{version_id}"]
    )

# --- Committer Agent ---
def commit_changes(files, version_id, message=None):
    # Step 1: Initialize repo if needed
    if not os.path.exists(".git"):
        print("No .git repository found. Initializing...")
        subprocess.run(["git", "init"])
        subprocess.run(["git", "checkout", "-b", "main"])

    # Step 2: Add files
    subprocess.run(["git", "add"] + files)

    # Step 3: Commit with version ID in message
    commit_message = message or f"Commit version {version_id}: EA refinement + logs"
    subprocess.run(["git", "commit", "-m", commit_message])

    # Step 4: Tag commit with version ID
    tag_name = f"v{version_id}"
    subprocess.run(["git", "tag", tag_name])

    # Step 5: Push commit and tag (requires remote configured)
    try:
        subprocess.run(["git", "push", "origin", "main"])
        subprocess.run(["git", "push", "origin", tag_name])
    except Exception as e:
        print("Push failed. Ensure remote 'origin' is configured:", e)

def get_artifacts_by_version(vectorstore, version_id):
    docs = vectorstore.similarity_search(f"version {version_id}", k=10)
    artifacts = {}
    for d in docs:
        artifacts[d.metadata.get("type")] = {
            "content": d.page_content,
            "metadata": d.metadata
        }
    return artifacts

def get_version_record(version_id):
    conn = sqlite3.connect("pipeline_dashboard.db")
    c = conn.cursor()
    c.execute("SELECT * FROM pipeline_versions WHERE version_id=?", (version_id,))
    record = c.fetchone()
    conn.close()
    return record

def record_pipeline_version(version_id, metrics):
    # Get latest commit hash
    commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    git_tag = f"v{version_id}"

    conn = sqlite3.connect("pipeline_dashboard.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_versions (
            version_id INTEGER PRIMARY KEY,
            git_commit_hash TEXT,
            git_tag TEXT,
            log_id TEXT,
            analysis_id TEXT,
            code_id TEXT,
            verdict_id TEXT,
            profit_factor REAL,
            drawdown REAL,
            win_rate REAL,
            timestamp TEXT
        )
    """)
    c.execute("""
        INSERT INTO pipeline_versions (
            version_id, git_commit_hash, git_tag,
            log_id, analysis_id, code_id, verdict_id,
            profit_factor, drawdown, win_rate, timestamp
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        version_id, commit_hash, git_tag,
        f"log_{version_id}", f"analysis_{version_id}",
        f"code_{version_id}", f"verdict_{version_id}",
        metrics["profit_factor"], metrics["drawdown"], metrics["win_rate"],
        datetime.now().isoformat()
    ))
    conn.commit()
    conn.close()

def run_pipeline(vectorstore, ollama_model, ea_file, header_files, set_file, log_text, metrics, ea_code, user_prompt=None):
    
    # Step 1: Generate new version ID
    version_id = generate_version_id(vectorstore)

    # Step 2: Store log
    store_log(vectorstore, log_text, metrics, version_id)

    # Step 3: Retrieve last two logs
    latest_log, previous_log = get_last_two_logs(vectorstore)

    # Step 4: Refine code with analysis + user guidance
    # refined_code, verdict = refine_code_with_analysis(
    #     ollama_model, latest_log, ea_code, user_prompt, previous_log, version_id
    # )

    refined_code, verdict, analysis_text = refine_code_with_analysis(
        ollama_model=ollama_model,
        latest_log=latest_log,
        ea_code=ea_code,
        user_prompt=user_prompt,
        previous_log=previous_log,
        version_id=version_id
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
            commit_changes([ea_file] + header_files + [set_file], version_id)
        else:
            print("No improvement detected. Commit skipped.")
    else:
        print("Baseline run — no previous log to compare yet.")

    # Step 6: Record dashboard entry (always)
    record_pipeline_version(version_id, metrics)
    print(f"Dashboard updated for version {version_id}")


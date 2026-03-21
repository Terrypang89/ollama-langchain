import subprocess
import os
import configparser
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import time
import psutil  # install with pip if needed
import shutil
import glob
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.documents import Document

load_dotenv()
OLLAMA_SERVER = os.getenv("OLLAMA_API_BASE")

class JSONMemory:
    def __init__(self, path="memory.json"):
        self.path = path
        if not os.path.exists(self.path):
            with open(self.path, "w") as f:
                json.dump({}, f)

    def store(self, key, value):
        data = self._load()
        data[key] = value
        self._save(data)

    def get(self, key):
        return self._load().get(key)

    def _load(self):
        with open(self.path, "r") as f:
            return json.load(f)

    def _save(self, data):
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

def compile_ea(mq5_file, metaeditor_path):
    """
    Compile an EA using MetaEditor.
    mq5_file: path to your .mq5 file
    metaeditor_path: path to MetaEditor.exe (usually in MT5 installation folder)
    """
    print("start compile EA ", mq5_file," ...")
    mq5_file = Path(mq5_file).resolve()
    cmd = [
        metaeditor_path,
        "/compile:" + str(mq5_file),
        "/log"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # print("Compiler stdout:", result.stdout)
    # print("Compiler stderr:", result.stderr)
    print("done comepile EA. ")
    # Check MetaEditor log file
    log_dir = Path(metaeditor_path).parent / "logs"
    log_file = log_dir / "MetaEditor.log"
    # if log_file.exists():
    #     print("MetaEditor log:")
    #     print(log_file.read_text(encoding="utf-8", errors="ignore"))

    return result.returncode == 0

def update_ini_file(
    ini_path,
    login,
    password,
    server,
    expert,
    symbol="XAUUSD",
    period="M5",
    from_date="2025.03.01",
    to_date="2025.04.01",
    deposit=10000,
    currency="USD",
    leverage="1:100",
    visual=0,
    report_path=r"C:\Users\Tofy3\Downloads\Tester_report.html",
    updates=None
):
    """
    Generate a tester.ini file with [Common], [Tester], and optional [TesterInputs].
    """
    if not os.path.exists(ini_path):
        ini_content = f"""[Common]
Login={login}
Password={password}
Server={server}
EnableNews=0

[Tester]
Expert={expert}
Symbol={symbol}
Period={period}
Optimization=0
Model=0
FromDate={from_date}
ToDate={to_date}
ForwardMode=0
Deposit={deposit}
Currency={currency}
ProfitInPips=0
Leverage={leverage}
ExecutionMode=0
OptimizationCriterion=0
Visual={visual}
ShutdownTerminal=true
ReplaceReport=true
Report={report_path}
"""
        with open(ini_path, "w") as f:
            f.write(ini_content)
        print("Generated new ini file:", ini_path)
    else:
        # Update existing ini file
        lines = []
        with open(ini_path, "r") as f:
            for line in f:
                updated = False
                if updates:
                    for key, value in updates.items():
                        if line.strip().startswith(f"{key}="):
                            lines.append(f"{key}={value}\n")
                            updated = True
                            break
                if not updated:
                    lines.append(line)
        with open(ini_path, "w") as f:
            f.writelines(lines)
        print("Updated existing ini file:", ini_path)

    return ini_path

def run_mt5_backtest(config_path, terminal_path, report_path, log_path, store_path,
                     portable_enable=True, timeout=60):
    """
    Run MT5 backtest via terminal command with timeout.
    Return (report_file, latest_log_file, archive_folder).
    """

    # --- Extract report path from ini file ---
    report_file = None
    with open(config_path, "r") as f:
        for line in f:
            if line.strip().startswith("Report="):
                report_file = line.strip().split("=", 1)[1]
                break

    if report_file and not os.path.isabs(report_file):
        report_file = os.path.join(report_path, report_file)

    # --- Cleanup old report variants ---
    if report_file:
        base_name = os.path.splitext(report_file)[0]
        for f in glob.glob(base_name + "*"):
            try:
                os.remove(f)
            except Exception:
                pass

    # --- Remove old log file ---
    latest_log_file = None
    if os.path.exists(log_path):
        log_files = [os.path.join(log_path, f) for f in os.listdir(log_path)
                     if os.path.isfile(os.path.join(log_path, f))]
        if log_files:
            latest_log_file = max(log_files, key=os.path.getmtime)
            try:
                os.remove(latest_log_file)
            except Exception:
                pass

    print("start backtesting using ", terminal_path ," ...")    
    # --- Build command ---
    cmd = [terminal_path, "/portable", f"/config:{config_path}"] if portable_enable \
          else [terminal_path, f"/config:{config_path}"]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()

    print("done ", timeout, " kill backtest ...")

    # --- Find latest log file after run ---
    latest_log_file = None
    if os.path.exists(log_path):
        log_files = [os.path.join(log_path, f) for f in os.listdir(log_path)
                     if os.path.isfile(os.path.join(log_path, f))]
        if log_files:
            latest_log_file = max(log_files, key=os.path.getmtime)

    # --- Create archive folder with new run_id ---
    today_str = datetime.today().strftime("%Y%m%d")
    existing = [d for d in os.listdir(store_path)
                if os.path.isdir(os.path.join(store_path, d)) and d.startswith(today_str)]
    version = 1
    if existing:
        versions = []
        for d in existing:
            parts = d.split("_")
            if len(parts) == 2 and parts[0] == today_str:
                try:
                    versions.append(int(parts[1]))
                except ValueError:
                    pass
        if versions:
            version = max(versions) + 1

    run_id = f"{today_str}_{version}"
    archive_folder = os.path.join(store_path, run_id)
    os.makedirs(archive_folder, exist_ok=True)

    # --- Copy artifacts ---
    if report_file and os.path.exists(report_file):
        base_name = os.path.splitext(report_file)[0]
        for f in glob.glob(base_name + "*"):
            if os.path.isfile(f):
                shutil.copy(f, archive_folder)
        if latest_log_file and os.path.exists(latest_log_file):
            shutil.copy(latest_log_file, archive_folder)

    return run_id, archive_folder, report_file, latest_log_file

# def parse_backtest_report(report_file):
#     """
#     Parse MT5 backtest HTML report into a DataFrame.
#     """
#     if not report_file or not os.path.exists(report_file):
#         raise FileNotFoundError(f"Report file not found or empty: {report_file}")

#     tables = pd.read_html(report_file)
#     if not tables:
#         raise ValueError("Report file is empty or contains no tables")

#     return tables[0]  

def parse_backtest_report(report_file):
    """
    Parse a MetaTrader backtest report file and extract summary metrics.
    Adjust parsing logic depending on whether it's HTML, XML, or CSV.
    """
    summary = {}
    with open(report_file, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Example: simple regex or string search
    # (replace with proper parsing depending on report format)
    if "Total trades" in content:
        summary["trades"] = int(content.split("Total trades")[1].split()[0])
    if "Profit" in content:
        summary["profit"] = float(content.split("Profit")[1].split()[0])
    if "Drawdown" in content:
        summary["drawdown"] = float(content.split("Drawdown")[1].split()[0])
    if "Profit Factor" in content:
        summary["profit_factor"] = float(content.split("Profit Factor")[1].split()[0])

    return summary

def load_params_from_ini(ini_file):
    """
    Load parameters from an INI file into a dict.
    """
    config = configparser.ConfigParser()
    config.read(ini_file)

    # Assume parameters are under a section called [Parameters]
    params = dict(config["Tester"])
    return params

def save_run_metadata(run_id, ea_name, parameters, archive_folder,
                      report_files, log_file, compiled_file, vector_ids, summary):
    """
    Save run metadata into metadata.json inside archive folder.
    Parameters are loaded from ini file.
    """
    metadata = {
        "run_id": run_id,
        "ea_name": ea_name,
        "parameters": parameters,   # dict from ini file
        "archive_folder": archive_folder,
        "report_files": report_files,
        "log_file": log_file,
        "compiled_file": compiled_file,
        "vector_ids": vector_ids,
        "summary": summary            # dict from parse_backtest_report
    }

    metadata_path = os.path.join(archive_folder, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved metadata.json for run {run_id}")

def save_run_and_update_memory(run_id, ea_name, parameters, archive_folder,
                               report_files, log_file, compiled_file,
                               summary_metrics, vector_ids):
    """
    Save metadata.json into archive_folder AND update memory.json with latest run info.
    Also append run details into a RUN_HISTORY list.
    """

    # --- Build metadata dictionary ---
    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "ea_name": ea_name,
        "parameters": parameters,
        "artifacts": {
            "archive_folder": archive_folder,
            "compiled_file": compiled_file,
            "report_variants": report_files,
            "log_file": log_file
        },
        "vector_db": vector_ids,
        "summary": summary_metrics
    }

    # --- Save metadata.json into archive folder ---
    os.makedirs(archive_folder, exist_ok=True)
    metadata_path = os.path.join(archive_folder, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Metadata saved to {metadata_path}")

    # --- Update memory.json ---
    ltm = JSONMemory(path="memory.json")

    # Store latest run info
    ltm.store("LAST_RUN_ID", run_id)
    ltm.store("LAST_RUN_EA", ea_name)
    ltm.store("LAST_RUN_ARCHIVE", archive_folder)

    for key, value in vector_ids.items():
        ltm.store(f"LAST_RUN_{key.upper()}", value)

    # Append to run history
    history = ltm.get("RUN_HISTORY") or []
    history.append(metadata)
    ltm.store("RUN_HISTORY", history)

    print(f"Updated memory.json with latest run {run_id} and appended to history.")

    return metadata_path

def embed_and_store(file_path, vector_db_path, doc_type, run_id):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([text])

    for d in docs:
        d.metadata = {"run_id": run_id, "doc_type": doc_type}

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=f"{OLLAMA_SERVER}"
    )

    # Ensure vector_db_path exists
    os.makedirs(vector_db_path, exist_ok=True)

    index_file = os.path.join(vector_db_path, "index.faiss")
    if os.path.exists(index_file):
        # Load existing FAISS index safely
        db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(docs)
    else:
        # Create new FAISS index from docs
        db = FAISS.from_documents(docs, embeddings)

    db.save_local(vector_db_path)

    return f"{doc_type}_{run_id}"

import os

def process_run_for_embeddings(run_id, archive_folder, ea_source_file, log_file, report_file, header_files=None):
    vector_db_path = "files_index"

    print("storing ea_source_file =", ea_source_file, "...")
    code_id = embed_and_store(ea_source_file, vector_db_path, "code", run_id)

    header_ids = []
    if header_files:
        for hf in header_files:
            print("storing header =", hf, "...")
            hid = embed_and_store(hf, vector_db_path, "header", run_id)
            header_ids.append(hid)

    log_id = None
    if log_file and os.path.exists(log_file):
        print("storing log_file =", log_file, "...")
        log_id = embed_and_store(log_file, vector_db_path, "log", run_id)
    else:
        print("log_file not found, skipping...")

    report_id = None
    if report_file and os.path.exists(report_file):
        print("storing report_file =", report_file, "...")
        report_id = embed_and_store(report_file, vector_db_path, "report", run_id)
    else:
        print("report_file not found, skipping...")

    print("done storing to vector at", vector_db_path)
    return {
        "code_embedding_id": code_id,
        "header_embedding_ids": header_ids,
        "log_embedding_id": log_id,
        "report_embedding_id": report_id
    }

def get_last_run_info():
    """Retrieve last run metadata from JSON-based memory."""
    ltm = JSONMemory(path="memory.json")
    return {
        "run_id": ltm.get("LAST_RUN_ID"),
        "archive_folder": ltm.get("LAST_RUN_ARCHIVE"),
        "ea_name": ltm.get("LAST_RUN_EA"),
        "vector_db": {
            "code_embedding_id": ltm.get("LAST_RUN_CODE_EMBEDDING_ID"),
            "log_embedding_id": ltm.get("LAST_RUN_LOG_EMBEDDING_ID"),
            "report_embedding_id": ltm.get("LAST_RUN_REPORT_EMBEDDING_ID"),
            "header_embedding_ids": ltm.get("LAST_RUN_HEADER_EMBEDDING_IDS")
        }
    }

def query_run_snippets(run_id, query_text, vector_db_path="files_index", top_k=3):
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=f"{OLLAMA_SERVER}"
    )
    db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
    results = db.similarity_search(query_text, k=top_k)
    return [r for r in results if r.metadata.get("run_id") == run_id]

def query_last_run_snippets(query_text, vector_db_path="files_index", top_k=3, doc_type=None):
    info = get_last_run_info()
    run_id = info["run_id"]

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
        base_url=f"{OLLAMA_SERVER}"
    )
    db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)

    results = db.similarity_search(query_text, k=top_k)
    filtered = [r for r in results if r.metadata.get("run_id") == run_id]
    if doc_type:
        filtered = [r for r in filtered if r.metadata.get("doc_type") == doc_type]
    return filtered


def suggest_code_improvements(query_text):
    # Step 1: Retrieve relevant snippets from latest run
    snippets = query_last_run_snippets(query_text)

    # Step 2: Build context for the model
    context = "\n\n".join([
        f"[{s.metadata.get('doc_type','unknown')} snippet]\n{s.page_content}"
        for s in snippets
    ])

    # Step 3: Send to model for improvement suggestions
    prompt = f"""
You are an expert MQL5 developer. Analyze the following snippets (code, headers, logs, reports)
and suggest improvements to the EA source code.

Query: {query_text}

Context:
{context}

Provide specific MQ5 code changes or refactoring ideas.
    """

    ollama_model = OllamaLLM(model="mistral:7b", base_url=OLLAMA_SERVER)
    response = ollama_model.invoke(prompt)

    return response

def apply_improvements_to_git(repo_dir, ea_file_path, header_file_path, improvements_text,
                              run_id=None,
                              commit_message="Apply MQ5 improvements",
                              mt5_mql5_folder=None):

    if not os.path.exists(repo_dir):
        os.makedirs(repo_dir)

    # Ensure repo exists
    if not os.path.exists(os.path.join(repo_dir, ".git")):
        print(f"No Git repo found in {repo_dir}. Initializing new repo...")
        subprocess.run(["git", "init"], cwd=repo_dir)

    # Append improvements
    with open(ea_file_path, "a", encoding="utf-8") as f:
        f.write("\n// --- Suggested Improvements ---\n")
        f.write("// " + improvements_text.replace("\n", "\n// ") + "\n")

    if header_file_path and os.path.exists(header_file_path):
        with open(header_file_path, "a", encoding="utf-8") as f:
            f.write("\n// --- Suggested Improvements ---\n")
            f.write("// " + improvements_text.replace("\n", "\n// ") + "\n")

    # Build commit message: run_id + ":" + commit_message
    if run_id:
        full_commit_message = f"{run_id}:{commit_message}"
    else:
        full_commit_message = commit_message

    # Commit changes
    subprocess.run(["git", "add", "."], cwd=repo_dir)
    subprocess.run(["git", "commit", "-m", full_commit_message], cwd=repo_dir)

    print(f"Improvements applied and committed: {full_commit_message}")

    # Tag commit with run_id if provided
    if run_id:
        subprocess.run(["git", "tag", run_id], cwd=repo_dir)
        print(f"Tagged commit with run_id: {run_id}")

    # Return the commit message string
    return full_commit_message

import os, json, subprocess

def record_git_commit_to_metadata_and_memory(run_id, commit_hash, repo_dir, archive_folder):
    """
    Record the latest Git commit hash into metadata.json and memory.json.
    """
    # Update metadata.json
    metadata_path = os.path.join(archive_folder, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    metadata["git_commit_hash"] = commit_hash
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # Update memory.json
    ltm = JSONMemory(path="memory.json")
    ltm.store("LAST_RUN_ID", run_id)
    ltm.store("LAST_RUN_GIT_COMMIT", commit_hash)

    print(f"Recorded commit {commit_hash} for run {run_id}")
    return commit_hash


def record_commit_to_metadata(run_id, archive_folder, commit_message, summary_metrics, vector_ids):
    """
    Record Git commit info into metadata.json in archive folder and update memory.json.
    """
    # Get latest commit hash
    repo_dir = os.path.dirname(archive_folder)  # adjust if repo lives elsewhere
    commit_hash = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_dir
    ).decode("utf-8").strip()

    # --- Update metadata.json in archive ---
    metadata_path = os.path.join(archive_folder, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    metadata["git_commit"] = commit_hash
    metadata["commit_message"] = commit_message
    metadata["run_id"] = run_id
    metadata["summary"] = summary_metrics
    metadata["vector_db"] = vector_ids

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Updated metadata.json with commit {commit_hash}")

    # --- Update memory.json ---
    ltm = JSONMemory(path="memory.json")
    ltm.store("LAST_RUN_ID", run_id)
    ltm.store("LAST_RUN_ARCHIVE", archive_folder)
    ltm.store("LAST_RUN_COMMIT", commit_hash)

    history = ltm.get("RUN_HISTORY") or []
    history.append(metadata)
    ltm.store("RUN_HISTORY", history)

    print(f"Updated memory.json with commit {commit_hash} for run {run_id}")


# compile EA
EA_NAME = os.getenv("EA_NAME")
EA_EX_NAME = os.getenv("EA_EX_NAME")
MT5_PATH = os.getenv("MT5_PATH")
AGENT_PATH = os.getenv("AGENT_PATH")

EA_MQ5_SUBPATH = os.getenv("EA_MQ5_SUBPATH")
EA_HEADER_SUBPATH = os.getenv("EA_HEADER_SUBPATH")
EA_MQL_FILE = os.path.join(AGENT_PATH, EA_MQ5_SUBPATH)
HEADER_MQL_FILE = os.path.join(AGENT_PATH, EA_HEADER_SUBPATH)
METAEDITOR = os.path.join(MT5_PATH, os.getenv("METAEDITOR_SUBPATH"))

TERMINAL_PATH = os.path.join(MT5_PATH, os.getenv("TERMINAL_SUBPATH"))
BACKTEST_REPORT_PATH = AGENT_PATH
BACKTEST_LOG_PATH = os.path.join(AGENT_PATH, os.getenv("BACKTEST_LOG_SUBPATH"))

print("EA_NAME:", EA_NAME)
print("EA_EX_NAME:", EA_EX_NAME)
print("MT5_PATH:", MT5_PATH)

print("AGENT_PATH:", AGENT_PATH)
print("EA_MQL_FILE:", EA_MQL_FILE)
print("HEADER_MQL_FILE:", HEADER_MQL_FILE)
print("METAEDITOR:", METAEDITOR)

print("TERMINAL_PATH:", TERMINAL_PATH)
print("BACKTEST_REPORT_PATH:", BACKTEST_REPORT_PATH)
print("BACKTEST_LOG_PATH:", BACKTEST_LOG_PATH)

result = compile_ea(EA_MQL_FILE, METAEDITOR)

#  generate ini
ini_file = update_ini_file(
    ini_path="tester.ini",
    login=os.getenv("MT5_LOGIN"),
    password=os.getenv("MT5_PASSWORD"),
    server=os.getenv("MT5_SERVER"),
    expert=EA_EX_NAME,
    symbol="XAUUSD",
    period="M5",
    from_date="2025.04.01",
    to_date="2025.04.05",
    report_path=r"Tester_report.html",
    updates={
        "Expert": EA_EX_NAME,
        "Symbol": "XAUUSD",
        "Period": "M5",
        "FromDate": "2025.05.01",
        "ToDate": "2025.05.10",
        "Visual" : False,
        "Report" : "Tester_report.html",
    }
)

# perform backtest
# report_file, log_file = run_mt5_backtest(ini_file, TERMINAL_PATH, BACKTEST_REPORT_PATH, BACKTEST_LOG_PATH, ".\logs", False, 30)
run_id, archive_folder, report_file, log_file = run_mt5_backtest(
    ini_file,
    TERMINAL_PATH,
    BACKTEST_REPORT_PATH,
    BACKTEST_LOG_PATH,
    "logs",
    False,
    30,
)
print("run_id:", run_id)
print("archive_folder:", archive_folder)
print("report_file:", report_file)
print("log_file:", log_file)

# generate vector id once stored to vector store
vector_ids = process_run_for_embeddings(
    run_id=run_id,
    archive_folder=archive_folder,
    ea_source_file=EA_MQL_FILE,
    log_file=log_file,
    report_file=report_file,
    header_files=[
        HEADER_MQL_FILE,
    ]
)

# Save metadata json first after backtest
save_run_metadata(
    run_id=run_id,
    ea_name=EA_NAME,
    parameters=load_params_from_ini(ini_file),
    archive_folder=archive_folder,
    report_files=[os.path.basename(report_file)],
    log_file=os.path.basename(log_file),
    compiled_file=EA_EX_NAME,
    vector_ids=vector_ids,
    summary=parse_backtest_report(report_file),
)

# merge metadata and memory json together
metadata_path = save_run_and_update_memory(
    run_id=run_id,
    ea_name=EA_NAME,
    parameters=load_params_from_ini(ini_file),
    archive_folder=archive_folder,
    report_files=[os.path.basename(report_file)],
    log_file=os.path.basename(log_file),
    compiled_file=EA_EX_NAME,
    summary_metrics=parse_backtest_report(report_file),
    vector_ids=vector_ids
)

code_suggest = "trailing stop logic with drawdown > 5%"
improvements = suggest_code_improvements(code_suggest)
print("Suggested Improvements:\n", improvements)

# Apply improvements and commit, returns the full commit message string
full_commit_message = apply_improvements_to_git(
    repo_dir="MQL5",
    ea_file_path=EA_MQ5_SUBPATH,
    header_file_path=EA_HEADER_SUBPATH,
    improvements_text=improvements,
    run_id=run_id,
    commit_message=code_suggest,
    mt5_mql5_folder=AGENT_PATH
)

# Get the commit hash of the commit just made
commit_hash = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd="MQL5"
).decode().strip()

# Record commit info into metadata.json and memory.json
record_git_commit_to_metadata_and_memory(run_id, commit_hash, "MQL5", archive_folder)




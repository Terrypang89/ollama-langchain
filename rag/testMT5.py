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
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, OllamaLLM
import codecs
import time

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

# Safe file read helper
def safe_read_file(path):
    try:
        # Try UTF-8 first
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        try:
            # Try UTF-16 (common for MT5 logs/reports)
            with open(path, "r", encoding="utf-16") as f:
                return f.read()
        except UnicodeDecodeError:
            # Fallback to latin-1 (Windows ANSI)
            with open(path, "r", encoding="latin-1") as f:
                return f.read()

def copyfiles(mq5_file, header_file, mql5_path):

    if os.path.exists(mq5_file) and os.path.exists(header_file):
        if os.path.isfile(mq5_file):
            mql5_ea_file = os.path.join(mql5_path, mq5_file)
            if os.path.exists(mql5_ea_file):
                shutil.copy(mq5_file, mql5_ea_file)
                print("copied ", mq5_file, " to ", mql5_ea_file)

        if os.path.isfile(header_file):
            mql5_header_file = os.path.join(mql5_path, header_file)
            if os.path.exists(mql5_header_file):
                shutil.copy(header_file, mql5_header_file)
                print("copied ", header_file, " to ", mql5_header_file)

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

    # --- Ensure store_path exists ---
    os.makedirs(store_path, exist_ok=True)

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

def report_tables_to_json(report_file, archive_folder, output_json="report_tables.json"):
    """
    Read all tables from a MetaTrader backtest HTML report and save them into a JSON file
    inside the archive folder.
    """
    if not os.path.exists(report_file):
        raise FileNotFoundError(f"Report file not found: {report_file}")

    tables = pd.read_html(report_file)
    if not tables:
        raise ValueError("No tables found in report")

    # Convert each DataFrame to a list of dicts
    tables_json = {}
    for idx, df in enumerate(tables):
        df.columns = [str(c).strip() for c in df.columns]
        tables_json[f"table_{idx}"] = df.to_dict(orient="records")

    # Ensure archive folder exists
    os.makedirs(archive_folder, exist_ok=True)

    # Build full path for output JSON file
    output_path = os.path.join(archive_folder, output_json)

    # Save to JSON file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tables_json, f, indent=4)

    print(f"Saved {len(tables)} tables to {output_path}")
    return output_path

def extract_number(val_str):
    """Return the first numeric value found in a string, or None."""
    match = re.search(r"-?\d+(\.\d+)?", val_str)
    if match:
        return float(match.group(0))
    return None

def beautify_report(data):
    """
    Beautify trading report JSON:
    - table_0: split into settings, inputs, results
    - table_1: restructure into table_orders and table_deals
    """

    def normalize_row(d):
        cleaned = {k: v for k, v in d.items() if v and str(v).lower() != "nan"}
        values = list(cleaned.values())
        if not values:
            return None
        unique_values = list(dict.fromkeys(values))
        if len(unique_values) == 1:
            return {"value": unique_values[0]}
        else:
            return {"label": unique_values[0], "value": unique_values[1]}

    def split_table0(lst):
        settings, inputs, results = [], [], []
        mode = "settings"
        for item in lst:
            if isinstance(item, dict):
                item = normalize_row(item)
            if not item:
                continue

            if mode == "settings":
                if item.get("label") == "Inputs:":
                    mode = "inputs"
                    inputs.append(item)
                    continue
                elif item.get("value") == "Results":
                    mode = "results"
                    continue
                settings.append(item)
            elif mode == "inputs":
                if item.get("value") == "Results":
                    mode = "results"
                    continue
                inputs.append(item)
            else:
                results.append(item)

        return settings, inputs, results

    def normalize_table1(table1):
        orders, deals = [], []
        order_headers, deal_headers = [], []
        mode = None

        for row in table1:
            if isinstance(row, dict):
                values = list(row.values())

                if "Orders" in values:
                    mode = "orders"
                    continue
                elif "Deals" in values:
                    mode = "deals"
                    continue

                if mode == "orders" and not order_headers:
                    order_headers = values
                    continue
                if mode == "deals" and not deal_headers:
                    deal_headers = values
                    continue

                if all(v is None or str(v).lower() == "nan" for v in values):
                    continue

                if mode == "orders" and order_headers:
                    row_dict = {order_headers[i]: values[i] if i < len(values) else None
                                for i in range(len(order_headers))}
                    orders.append(row_dict)
                elif mode == "deals" and deal_headers:
                    row_dict = {deal_headers[i]: values[i] if i < len(values) else None
                                for i in range(len(deal_headers))}
                    deals.append(row_dict)

        return orders, deals

    # Apply transformations
    if "table_0" in data:
        settings, inputs, results = split_table0(data["table_0"])
        data["table_0"] = settings
        if inputs:
            data["table_inputs"] = inputs
        if results:
            data["table_results"] = results
    if "table_1" in data:
        orders, deals = normalize_table1(data["table_1"])
        data["table_orders"] = orders
        data["table_deals"] = deals
        del data["table_1"]

    return data

def extract_tester_report_summary(json_file):
    """
    Load a tester report JSON, clean it with beautify_report,
    save the cleaned version, remove unnecessary tables,
    and delete the original file.

    Args:
        json_file (str or Path): Path to the raw tester report JSON.

    Returns:
        tuple: (cleaned_data dict, Path to cleaned JSON file)
    """
    json_file = Path(json_file)

    # Load raw JSON
    with json_file.open("r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Apply cleaning
    cleaned_data = beautify_report(raw_data)

    # Save cleaned report
    report_tables_clean_file = json_file.parent / "report_tables_clean.json"
    with report_tables_clean_file.open("w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=4, ensure_ascii=False)
    print(f"Cleaned report saved to: {report_tables_clean_file}")

    # Remove unnecessary tables if present
    for key in ("table_0", "table_inputs"):
        if key in cleaned_data:
            del cleaned_data[key]

    # Delete original file safely
    if json_file.exists():
        json_file.unlink()
        print(f"Original file removed: {json_file}")

    return cleaned_data, report_tables_clean_file


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
                      report_files, log_file, vector_ids, summary):
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
        "vector_ids": vector_ids,
        "summary": summary            # dict from parse_backtest_report
    }

    metadata_path = os.path.join(archive_folder, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved metadata.json for run {run_id}")

def save_run_and_update_memory(run_id, parameters, archive_folder,
                               report_files, log_file, vector_ids):
    """
    Save run metadata into metadata.json and update memory.json with latest run info.
    """
    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "parameters": parameters,
        "artifacts": {
            "archive_folder": archive_folder,
            "report_variants": report_files,
            "log_file": log_file,
        },
        "vector_db": vector_ids,
    }

    # Save metadata.json inside archive folder
    metadata_path = os.path.join(archive_folder, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # Update memory.json
    ltm = JSONMemory(path="memory.json")
    ltm.store("LAST_RUN_ID", run_id)
    ltm.store("LAST_RUN_ARCHIVE", archive_folder)
    ltm.store("LAST_RUN_CODE_EMBEDDING_ID", vector_ids.get("code_embedding_id"))
    ltm.store("LAST_RUN_HEADER_EMBEDDING_ID", vector_ids.get("header_embedding_id"))
    ltm.store("LAST_RUN_LOG_EMBEDDING_ID", vector_ids.get("log_embedding_id"))
    ltm.store("LAST_RUN_REPORT_EMBEDDING_ID", vector_ids.get("report_embedding_id"))

    # Insert newest run at the top of history
    history = ltm.get("RUN_HISTORY") or []
    history.insert(0, metadata)   # instead of append
    ltm.store("RUN_HISTORY", history)

    print(f"Saved run {run_id} and updated memory.json")

def embed_and_store(ollama_server, file_path, vector_db_path, doc_type, run_id,
                    chunk_size=2000, chunk_overlap=50):
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.create_documents([text])

    for i, d in enumerate(docs):
        d.metadata = {"run_id": run_id, "doc_type": doc_type, "chunk_index": i}

    embeddings = OllamaEmbeddings(
        model=os.getenv("EMBEDDED_AGENT"),
        base_url=f"{ollama_server}"
    )

    os.makedirs(vector_db_path, exist_ok=True)
    index_file = os.path.join(vector_db_path, "index.faiss")

    if os.path.exists(index_file):
        db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)
        db.add_documents(docs)
    else:
        db = FAISS.from_documents(docs, embeddings)

    db.save_local(vector_db_path)
    return f"{doc_type}_{run_id}"

def process_run_for_embeddings(
    ollama_server, run_id,
    ea_source_file, log_file, report_file,
    header_files=None,
    log_chunk_size=10000   # number of lines per chunk
):
    vector_db_path = "files_index"

    def store_with_stats(file_path, doc_type):
        if file_path and os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"storing {doc_type} = {file_path} (size: {size_mb:.2f} MB)...")
            start = time.perf_counter()
            embedding_id = embed_and_store(ollama_server, file_path, vector_db_path, doc_type, run_id)
            elapsed = time.perf_counter() - start
            print(f"{doc_type} stored in {elapsed:.2f} seconds")
            return embedding_id
        else:
            print(f"{doc_type} file not found, skipping...")
            return None

    # EA source file
    code_id = store_with_stats(ea_source_file, "code")

    # Header file(s) – if you want multiple headers, loop here
    header_id = store_with_stats(header_files, "header")

    # Log file with chunking
    log_id = None
    if log_file and os.path.exists(log_file):
        size_mb = os.path.getsize(log_file) / (1024 * 1024)
        print(f"storing log_file = {log_file} (size: {size_mb:.2f} MB)...")
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        chunk_ids = []
        for i in range(0, len(lines), log_chunk_size):
            chunk_lines = lines[i:i+log_chunk_size]
            chunk_file = f"{log_file}_part_{i//log_chunk_size}.txt"
            with open(chunk_file, "w", encoding="utf-8") as out:
                out.writelines(chunk_lines)

            start = time.perf_counter()
            chunk_id = embed_and_store(ollama_server, chunk_file, vector_db_path, "log", run_id)
            elapsed = time.perf_counter() - start
            print(f"log chunk {i//log_chunk_size} stored in {elapsed:.2f} seconds")
            chunk_ids.append(chunk_id)

        # unify into a single identifier
        log_id = f"log_{run_id}"
    else:
        print("log_file not found, skipping...")

    # Report file
    report_id = store_with_stats(report_file, "report")

    print("done storing to vector at", vector_db_path)
    return {
        "code_embedding_id": code_id,
        "header_embedding_id": header_id,
        "log_embedding_id": log_id,
        "report_embedding_id": report_id
    }

def get_last_run_info():
    """Retrieve last run metadata from JSON-based memory."""
    ltm = JSONMemory(path="memory.json")
    return {
        "run_id": ltm.get("LAST_RUN_ID"),
        "vector_db": {
            "code_embedding_id": ltm.get("LAST_RUN_CODE_EMBEDDING_ID"),
            "log_embedding_id": ltm.get("LAST_RUN_LOG_EMBEDDING_ID"),
            "report_embedding_id": ltm.get("LAST_RUN_REPORT_EMBEDDING_ID"),
            "header_embedding_id": ltm.get("LAST_RUN_HEADER_EMBEDDING_ID")  # singular for consistency
        }
    }

def query_last_run_snippets(
    ollama_server,
    query_text=None,
    vector_db_path="files_index",
    top_k=10,
    doc_type=None
):
    """Query FAISS index for snippets from the last run, optionally filtered by doc_type."""
    info = get_last_run_info()
    run_id = info["run_id"]
    print("get_last_run_info:", info)

    embeddings = OllamaEmbeddings(
        model=os.getenv("EMBEDDED_AGENT"),
        base_url=f"{ollama_server}"
    )
    db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)

    # If query_text provided, do similarity search
    if query_text:
        results = db.similarity_search(query_text, k=top_k)
        filtered = [r for r in results if r.metadata.get("run_id") == run_id]
    else:
        print("No query_text so pull all docs")
        # Directly pull all docs for this run
        all_docs = list(db.docstore._dict.values())
        filtered = [r for r in all_docs if r.metadata.get("run_id") == run_id]

        # Limit results if top_k specified
        if top_k and len(filtered) > top_k:
            filtered = filtered[:top_k]

    # Optional filter by doc_type
    if doc_type:
        filtered = [r for r in filtered if r.metadata.get("doc_type") == doc_type]

    return filtered

def analyze_and_improve(ollama_server, query_text=None):

    start = time.perf_counter()
    print("start analyze_and_improve")
    snippets = query_last_run_snippets(ollama_server, query_text)
    elapsed = time.perf_counter() - start
    # print("last_run_snippets:",snippets , " with {elapsed:.2f} seconds")

    reasoning_prompt = f"""
    You are an expert MQL5 developer and quantitative trader.
    Analyze the loss deals, resolve with EA improvements, retest, and check differences.

    Tasks:
    1) Identify all loss deals in report_tables_clean table_deals from context.
    2) For each loss deal, match the deal time with context log entries and extract the reason.
    3) Inspect the EA function Trade_Strategy. Advise how to resolve the issue.
    4) Propose specific MQL5 code changes or refactors in Trade_Strategy (show code snippets).
    5) Compare the new backtest results. Verify if the previously losing deals are now resolved.

    Context:
    { "\n\n".join([s.page_content for s in snippets]) }
    """
    print("perform reasoning_prompt for analysis.")
    start = time.perf_counter()
    reasoning_model = OllamaLLM(model=os.getenv("REASONING_AGENT"), base_url=ollama_server)
    analysis = reasoning_model.invoke(reasoning_prompt)
    elapsed = time.perf_counter() - start
    print("analysis:", analysis, " with {elapsed:.2f} seconds")

    coder_prompt = f"Based on this analysis:\n{analysis}\nShow MQL5 code changes in Trade_Strategy."
    print("coder_prompt:", coder_prompt)
    start = time.perf_counter()
    coder_model = OllamaLLM(model=os.getenv("CODER_AGENT"), base_url=ollama_server)
    code_fix = coder_model.invoke(coder_prompt)
    elapsed = time.perf_counter() - start
    print("code_fix:", code_fix, " with {elapsed:.2f} seconds")
    return {"reasoning_prompt": reasoning_prompt, "analysis": analysis, "code_fix": code_fix}


def suggest_code_improvements(ollama_server, query_text):
    # Step 1: Retrieve relevant snippets from latest run
    snippets = query_last_run_snippets(ollama_server, query_text)

    # Step 2: Build context for the model
    context = "\n\n".join([
        f"[{s.metadata.get('doc_type','unknown')} snippet]\n{s.page_content}"
        for s in snippets
    ])

    # Step 3: Send to model for improvement suggestions
    prompt = f"""
You are an expert MQL5 developer. Analyze the following snippets (code, headers, logs, reports)
and suggest improvements to the EA source code. Specially trading strategy. 

Query: {query_text}

Context:
{context}

Provide specific MQ5 code changes or refactoring ideas.
    """

    ollama_model = OllamaLLM(
        model=os.getenv("CODER_AGENT"),
        base_url=OLLAMA_SERVER
    )
    response = ollama_model.invoke(prompt)

    return response

def generate_patch_from_git(code_repo, old_code_file, user_commit_message, new_code):
    """
    Overwrite the file with new code, stage changes, commit, generate a patch file,
    then reset back to baseline. Returns the full path to the generated patch file.
    """

    try:
        # --- Ensure repo is initialized ---
        if not os.path.exists(os.path.join(code_repo, ".git")):
            subprocess.run(["git", "init"], cwd=code_repo, check=True)
            gitignore_path = os.path.join(code_repo, ".gitignore")
            if not os.path.exists(gitignore_path):
                with open(gitignore_path, "w", encoding="utf-8") as gi:
                    gi.write("*.patch\n")
                    gi.write("patches\n")
            subprocess.run(["git", "add", "."], cwd=code_repo, check=True)
            subprocess.run(["git", "commit", "-m", "init with files"], cwd=code_repo, check=True)

        # --- Overwrite file with regenerated code ---
        with open(old_code_file, "w", encoding="utf-8") as f:
            f.write(new_code)

        # --- Stage the file (relative path from repo root) ---
        rel_path = os.path.relpath(old_code_file, code_repo)
        subprocess.run(["git", "add", rel_path], cwd=code_repo, check=True)

        # --- Check if there are staged changes ---
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=code_repo,
            capture_output=True,
            text=True
        )
        if not status.stdout.strip():
            raise RuntimeError("No changes to commit. Did you overwrite the file with new content?")

        # --- Commit ---
        subprocess.run(["git", "commit", "-m", user_commit_message], cwd=code_repo, check=True)

        # --- Generate patch file ---
        patch_dir = os.path.join(code_repo, "patches")
        os.makedirs(patch_dir, exist_ok=True)

        result = subprocess.run(
            ["git", "format-patch", "-1", "HEAD", "-o", "patches"],
            cwd=code_repo,
            check=True,
            capture_output=True,
            text=True
        )
        patch_filename = result.stdout.strip()
        patch_path = os.path.join(patch_dir, patch_filename)

        # --- Reset back to baseline ---
        subprocess.run(["git", "reset", "--hard", "HEAD~1"], cwd=code_repo, check=True)

        return patch_path

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Git error while generating patch: {e}") from e


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

        # wheck file path exist
        if not os.path.exists(os.path.dirname(ea_file_path)):
            os.makedirs(os.path.dirname(ea_file_path))
            mql5_folder_ea_file = os.path.join(mt5_mql5_folder, ea_file_path)
            print("copy files:", mql5_folder_ea_file, " to ", ea_file_path)
            if os.path.isfile(mql5_folder_ea_file):
                shutil.copy(mql5_folder_ea_file, ea_file_path)

        if not os.path.exists(os.path.dirname(header_file_path)):
            os.makedirs(os.path.dirname(header_file_path))
            mql5_folder_header_file = os.path.join(mt5_mql5_folder, header_file_path)
            print("copy files:", mql5_folder_header_file, " to ", header_file_path)
            if os.path.isfile(mql5_folder_header_file):
                shutil.copy(mql5_folder_header_file, header_file_path)

        if os.path.exists(ea_file_path) and os.path.exists(header_file_path):
            commit_message = "init with " + ea_file_path + " && " + header_file_path
        else:
            commit_message = " init with none."

    # Append improvements
    if ea_file_path and os.path.exists(ea_file_path):
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

def clean_log(input_file: str, archive_folder: str = "archive") -> str:
    """
    Remove the first 4 whitespace-separated columns from each line
    in the log file and write to a new file inside the archive folder.
    Returns the path of the cleaned output file.
    """
    # Ensure archive folder exists
    os.makedirs(archive_folder, exist_ok=True)

    # Build output filename inside archive folder
    base_name = os.path.basename(input_file)
    name, ext = os.path.splitext(base_name)
    output_file = os.path.join(archive_folder, f"{name}_clean{ext}")

    # Try common encodings until one works
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            with open(input_file, "r", encoding=enc) as infile, open(output_file, "w", encoding="utf-8") as outfile:
                for line in infile:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cleaned = " ".join(parts[4:])
                    outfile.write(cleaned + "\n")
            print(f"Cleaned log written to {output_file} (read as {enc})")
            return output_file
        except UnicodeDecodeError:
            continue

    raise ValueError("Failed to decode file with common encodings.")

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

load_dotenv()
OLLAMA_SERVER = os.getenv("OLLAMA_API_BASE")

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
print("start operation =============================================================================")

# copyfiles(EA_MQ5_SUBPATH, EA_HEADER_SUBPATH, AGENT_PATH)

# result = compile_ea(EA_MQL_FILE, METAEDITOR)

# #  generate ini
# ini_file = update_ini_file(
#     ini_path="tester.ini",
#     login=os.getenv("MT5_LOGIN"),
#     password=os.getenv("MT5_PASSWORD"),
#     server=os.getenv("MT5_SERVER"),
#     expert=EA_EX_NAME,
#     symbol="XAUUSD",
#     period="M5",
#     from_date="2025.04.01",
#     to_date="2025.04.05",
#     report_path=r"Tester_report.html",
#     updates={
#         "Expert": EA_EX_NAME,
#         "Symbol": "XAUUSD",
#         "Period": "M5",
#         "FromDate": "2025.05.01",
#         "ToDate": "2025.05.10",
#         "Visual" : False,
#         "Report" : "Tester_report.html",
#     }
# )

# # perform backtest
# run_id, archive_folder, report_file, log_file = run_mt5_backtest(
#     ini_file,
#     TERMINAL_PATH,
#     BACKTEST_REPORT_PATH,
#     BACKTEST_LOG_PATH,
#     "logs",
#     False,
#     30,
# )
# print("run_id:", run_id)
# print("archive_folder:", archive_folder)
# print("report_file:", report_file)
# print("log_file:", log_file)

# json_report_file = report_tables_to_json(
#     report_file, 
#     archive_folder=archive_folder,
#     output_json="report_tables.json",
# )

# extract_Tester_report_summary(json_report_file)

# # generate vector id once stored to vector store
# vector_ids = process_run_for_embeddings(
#     ollama_server=OLLAMA_SERVER,
#     run_id=run_id,
#     archive_folder=archive_folder,
#     ea_source_file=EA_MQL_FILE,
#     log_file=log_file,
#     report_file=report_file,
#     header_files=[
#         HEADER_MQL_FILE,
#     ]
# )

# # merge metadata and memory json together
# metadata_path = save_run_and_update_memory(
#     run_id=run_id,
#     ea_name=EA_NAME,
#     parameters=load_params_from_ini(ini_file),
#     archive_folder=archive_folder,
#     report_files=[os.path.basename(report_file)],
#     log_file=os.path.basename(log_file),
#     compiled_file=EA_EX_NAME,
#     vector_ids=vector_ids,
#     summary=parse_backtest_report(json_report_file),
# )

# code_suggest = "trailing stop logic with drawdown > 5%"
# improvements = suggest_code_improvements(OLLAMA_SERVER, code_suggest)
# # print("Suggested Improvements:\n", improvements)

# # Apply improvements and commit, returns the full commit message string
# full_commit_message = apply_improvements_to_git(
#     repo_dir="MQL5",
#     ea_file_path=EA_MQ5_SUBPATH,
#     header_file_path=EA_HEADER_SUBPATH,
#     improvements_text=improvements,
#     run_id=run_id,
#     commit_message=code_suggest,
#     mt5_mql5_folder=AGENT_PATH
# )

# # Get the commit hash of the commit just made
# commit_hash = subprocess.check_output(
#     ["git", "rev-parse", "HEAD"], cwd="MQL5"
# ).decode().strip()

# # Record commit info into metadata.json and memory.json
# record_git_commit_to_metadata_and_memory(run_id, commit_hash, "MQL5", archive_folder)




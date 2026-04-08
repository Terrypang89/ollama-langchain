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
from typing import List
import tiktoken

# Initialize tokenizer (cl100k_base works well for LLaMA‑style models)
enc = tiktoken.get_encoding("cl100k_base")

MAX_TOKENS = 32768


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

def beautify_text(text) -> str:
    """
    Beautify analysis whether it's a string or a list of strings.
    - Joins lists into one string
    - Splits into lines
    - Adds bullet points for readability
    - Preserves code blocks if present
    """
    if not text:
        return ""

    # If it's a list, join items into one string
    if isinstance(text, list):
        text = "\n".join(str(t) for t in text if t)

    # If it's not already a string, force conversion
    text = str(text)

    # Preserve code blocks: split by triple backticks
    if "```" in text:
        parts = text.split("```")
        beautified_parts = []
        for i, part in enumerate(parts):
            if i % 2 == 1:  # inside code block
                beautified_parts.append("```" + part.strip() + "```")
            else:  # normal text
                lines = [f"- {line.strip()}" for line in part.split("\n") if line.strip()]
                beautified_parts.append("\n".join(lines))
        return "\n\n".join(beautified_parts)

    # Otherwise, just bulletize plain text
    lines = [f"- {line.strip()}" for line in text.split("\n") if line.strip()]
    return "\n".join(lines)

def count_tokens(data, model_name="gpt-3.5-turbo", verbose=False):
    """Count tokens for either a string or a list of snippet objects."""
    try:
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback for non-OpenAI models (like Qwen, LLaMA, etc.)
        enc = tiktoken.get_encoding("cl100k_base")

    # Case 1: data is a string
    if isinstance(data, str):
        tokens = len(enc.encode(data))
        if verbose:
            print(f"String token count: {tokens}")
        return tokens

    # Case 2: data is a list of snippets
    elif isinstance(data, (list, tuple)):
        total_tokens = 0
        for idx, s in enumerate(data, start=1):
            # Handle snippet objects with .page_content or plain strings
            text = getattr(s, "page_content", s)  
            tokens = len(enc.encode(text))
            total_tokens += tokens
            if verbose:
                section = getattr(s, "section", "unknown")
                print(f"Snippet {idx} ({section}): {tokens} tokens")
        return total_tokens

    else:
        raise TypeError("Unsupported type for count_tokens: must be str or list of snippets")


def chunk_prompt(prompt: str, max_tokens: int = MAX_TOKENS) -> List[str]:
    """Split a long prompt into chunks that fit within the token limit."""
    tokens = enc.encode(prompt)
    chunks = []
    for i in range(0, len(tokens), max_tokens):
        chunk = tokens[i:i+max_tokens]
        chunks.append(enc.decode(chunk))
    return chunks

def safe_invoke_ollama(prompt: str, model_name: str, base_url: str):
    """Count tokens, chunk if needed, and sequentially invoke Ollama."""
    token_count = len(enc.encode(prompt))
    print(f"Prompt tokens: {token_count}")

    coder_model = OllamaLLM(model=model_name, base_url=base_url)

    if token_count <= MAX_TOKENS:
        # Safe to send directly
        return coder_model.invoke(prompt)
    else:
        print(f"Prompt exceeds {MAX_TOKENS} tokens, chunking...")
        responses = []
        for idx, chunk in enumerate(chunk_prompt(prompt)):
            start = time.perf_counter()
            print(f"Sending chunk {idx+1}/{len(chunk_prompt(prompt))} "
                  f"({len(enc.encode(chunk))} tokens)")
            resp = coder_model.invoke(chunk)
            elapsed = time.perf_counter() - start
            print(f"Elapsed for chunk {idx+1}: {elapsed:.2f} seconds")
            responses.append(resp)
        return responses

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
    Return (report_file, latest_log_file, archieve_folder).
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

    # --- Create archieve folder with new run_id ---
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
    archieve_folder = os.path.join(store_path, run_id)
    os.makedirs(archieve_folder, exist_ok=True)

    # --- Copy artifacts ---
    if report_file and os.path.exists(report_file):
        base_name = os.path.splitext(report_file)[0]
        for f in glob.glob(base_name + "*"):
            if os.path.isfile(f):
                shutil.copy(f, archieve_folder)
        if latest_log_file and os.path.exists(latest_log_file):
            shutil.copy(latest_log_file, archieve_folder)

    return run_id, archieve_folder, report_file, latest_log_file

def report_tables_to_json(report_file, archieve_folder, output_json="report_tables.json"):
    """
    Read all tables from a MetaTrader backtest HTML report and save them into a JSON file
    inside the archieve folder.
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

    # Ensure archieve folder exists
    os.makedirs(archieve_folder, exist_ok=True)

    # Build full path for output JSON file
    output_path = os.path.join(archieve_folder, output_json)

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

def save_run_metadata(run_id, ea_name, parameters, archieve_folder,
                      report_files, log_file, vector_ids, summary):
    """
    Save run metadata into metadata.json inside archieve folder.
    Parameters are loaded from ini file.
    """
    metadata = {
        "run_id": run_id,
        "ea_name": ea_name,
        "parameters": parameters,   # dict from ini file
        "archieve_folder": archieve_folder,
        "report_files": report_files,
        "log_file": log_file,
        "vector_ids": vector_ids,
        "summary": summary            # dict from parse_backtest_report
    }

    metadata_path = os.path.join(archieve_folder, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved metadata.json for run {run_id}")

def save_run_and_update_memory(run_id, parameters, archieve_folder,
                               report_files, log_file, vector_ids):
    """
    Save run metadata into metadata.json and update memory.json with latest run info.
    """
    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "parameters": parameters,
        "artifacts": {
            "archieve_folder": archieve_folder,
            "report_variants": report_files,
            "log_file": log_file,
        },
        "vector_db": vector_ids,
    }

    # Save metadata.json inside archieve folder
    metadata_path = os.path.join(archieve_folder, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # Update memory.json
    ltm = JSONMemory(path="memory.json")
    ltm.store("LAST_RUN_ID", run_id)
    ltm.store("LAST_RUN_ARCHIEVE", archieve_folder)
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

    section_counts = {}
    file_path_str = str(file_path)

    for i, d in enumerate(docs):
        # Section tagging
        section = "general"
        if "table_deals" in d.page_content:
            section = "table_deals"
        elif "table_orders" in d.page_content:
            section = "table_orders"
        elif "table_results" in d.page_content:
            section = "table_results"
        elif "Trade_Strategy" in d.page_content or file_path_str.endswith(".mq5"):
            section = "mq5_code"
        elif file_path_str.endswith(".mqh"):
            section = "mqh_header"
        elif file_path_str.endswith(".log") or (file_path_str.endswith(".txt") and "log_part" in file_path_str):
            section = "log"

        d.metadata = {
            "run_id": run_id,
            "doc_type": doc_type,
            "chunk_index": i,
            "section": section
        }

        # Count chunks per section
        section_counts[section] = section_counts.get(section, 0) + 1

    # Debug printout
    print(f"Embedded {len(docs)} chunks from {file_path}")
    for section, count in section_counts.items():
        print(f"  {section}: {count} chunks")

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
            print(f"{doc_type} stored in {elapsed:.2f} seconds (embedding_id={embedding_id})")
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
        "archieve": ltm.get("LAST_RUN_ARCHIEVE"),
        "vector_db": {
            "code_embedding_id": ltm.get("LAST_RUN_CODE_EMBEDDING_ID"),
            "log_embedding_id": ltm.get("LAST_RUN_LOG_EMBEDDING_ID"),
            "report_embedding_id": ltm.get("LAST_RUN_REPORT_EMBEDDING_ID"),
            "header_embedding_id": ltm.get("LAST_RUN_HEADER_EMBEDDING_ID")  # singular for consistency
        }
    }

def query_last_run_snippets(ollama_server, doc_type=None, section=None, top_k=10, query_text=None):
    info = get_last_run_info()
    run_id = info["run_id"]
    vector_db_path="files_index"

    embeddings = OllamaEmbeddings(
        model=os.getenv("EMBEDDED_AGENT"),
        base_url=f"{ollama_server}"
    )
    db = FAISS.load_local(vector_db_path, embeddings, allow_dangerous_deserialization=True)

    if query_text:
        results = db.similarity_search(query_text, k=top_k)
        filtered = [r for r in results if r.metadata.get("run_id") == run_id]
    else:
        all_docs = list(db.docstore._dict.values())
        filtered = [r for r in all_docs if r.metadata.get("run_id") == run_id]

    if doc_type:
        filtered = [r for r in filtered if r.metadata.get("doc_type") == doc_type]

    if section:
        filtered = [r for r in filtered if r.metadata.get("section") == section]

    if top_k and len(filtered) > top_k:
        filtered = filtered[:top_k]

    return filtered, info["archieve"], run_id

def inspect_snippets(snippets, expected_sections=None):
    print(f"Total snippets retrieved: {len(snippets)}")

    # Count by section
    section_counts = {}
    for s in snippets:
        section = s.get("section", "unknown")
        section_counts[section] = section_counts.get(section, 0) + 1

    print("\nCounts per section:")
    for section, count in section_counts.items():
        print(f"  {section}: {count}")

    # Warn if expected sections are missing
    if expected_sections:
        for section in expected_sections:
            if section not in section_counts:
                print(f"⚠️ Warning: No snippets found for section '{section}'")

    # Preview each snippet
    for i, s in enumerate(snippets):
        print(f"\n--- Snippet {i} ---")
        print("Section:", s.get("section", "unknown"))
        # tokens are optional now, so only show if present
        if "tokens" in s:
            print("Tokens:", s["tokens"])
        print("Preview:", s.get("content", "")[:200].replace("\n", " "))

def store_snippets(ollama_server, model_name):
    # Retrieve snippets
    deals_snippets, archieve_dir, run_id = query_last_run_snippets(ollama_server, "report", "table_deals", 100)
    orders_snippets, archieve_dir, run_id = query_last_run_snippets(ollama_server, "report", "table_orders", 100)
    results_snippets, archieve_dir, run_id = query_last_run_snippets(ollama_server, "report", "table_results", 100)
    code_snippets, archieve_dir, run_id = query_last_run_snippets(ollama_server, "code", "mq5_code", 10)
    header_snippets, archieve_dir, run_id = query_last_run_snippets(ollama_server, "header", "mqh_header", 10)
    log_snippets, archieve_dir, run_id = query_last_run_snippets(ollama_server, "log", "log", 10)

    def extract_content(snippet_list):
        return [getattr(s, "page_content", str(s)) for s in snippet_list]

    # Build per-section dicts
    json_snippets = {
        "table_deals": {
            "content": extract_content(deals_snippets),
            "total_tokens": count_tokens(extract_content(deals_snippets), model_name, False)
        },
        "table_orders": {
            "content": extract_content(orders_snippets),
            "total_tokens": count_tokens(extract_content(orders_snippets), model_name, False)
        },
        "table_results": {
            "content": extract_content(results_snippets),
            "total_tokens": count_tokens(extract_content(results_snippets), model_name, False)
        },
        "mq5_code": {
            "content": extract_content(code_snippets),
            "total_tokens": count_tokens(extract_content(code_snippets), model_name, False)
        },
        "mqh_header": {
            "content": extract_content(header_snippets),
            "total_tokens": count_tokens(extract_content(header_snippets), model_name, False)
        },
        "log": {
            "content": extract_content(log_snippets),
            "total_tokens": count_tokens(extract_content(log_snippets), model_name, False)
        }
    }

    # Compute overall total
    all_texts = []
    for section in json_snippets.values():
        all_texts.extend(section["content"])
    total_tokens = count_tokens(all_texts, model_name, False)

    # Save JSON
    if archieve_dir and os.path.exists(archieve_dir):
        archieve_file = os.path.join(archieve_dir, f"snippets_{run_id}.json")
        with open(archieve_file, "w", encoding="utf-8") as f:
            json.dump({
                "snippets": json_snippets,
                "total_tokens": total_tokens
            }, f, indent=2)
        print(f"✅ Stored latest snippets with id {run_id} to {archieve_file}")
        return archieve_file
    else:
        print("⚠️ Archive directory not found.")
        return None

def load_snippets(ollama_server, model_name):
    info = get_last_run_info()
    run_id = info["run_id"]
    archieve_dir = info["archieve"]
    archieve_file = os.path.join(archieve_dir, f"snippets_{run_id}.json")
    if not os.path.exists(archieve_file) or not archieve_file:
        print(f"⚠️ No archieve found for run_id {run_id}, creating one...")
        archieve_file = store_snippets(ollama_server, model_name)

    if archieve_file and os.path.exists(archieve_file):
        with open(archieve_file, "r", encoding="utf-8") as f:
            json_snippets = json.load(f)

        return json_snippets, archieve_dir, run_id
    else:
        print("❌ Failed to create snippet archieve.")
        return {}, archieve_dir, run_id

def analyze_and_improve(ollama_server, user_prompt, snippets_enable=False, query_text=None):
    start = time.perf_counter()
    model_name = os.getenv("REASONING_AGENT")
    print("start analyze_and_improve")

    # Load snippets from archieve (dicts with content/tokens/section)
    data, archieve_dir, run_id = load_snippets(ollama_server, model_name)

    snippets_json = data.get("snippets", {})
    total_tokens = data.get("total_tokens", 0)

    print(f"Overall total tokens: {total_tokens}")

    for section_name, section_data in snippets_json.items():
        section_tokens = section_data.get("total_tokens")
        if section_tokens is not None:
            print(f"Section {section_name} tokens: {section_tokens}")
        else:
            print(f"Section {section_name} has {len(section_data.get('content', []))} snippets")

    # Flatten into one list of texts
    all_texts = []
    for section_name, section_data in snippets_json.items():
        all_texts.extend(section_data.get("content", []))

    elapsed = time.perf_counter() - start
    print(f"retrieved {len(all_texts)} snippets with Total tokens of {total_tokens} in {elapsed:.2f} seconds")

    # Inspect snippets AFTER retrieval (adapted for dict format)
    inspect_snippets(
        [{"section": sec, "content": txt}
        for sec, sec_data in snippets_json.items()
        for txt in sec_data.get("content", [])],
        expected_sections=["table_deals", "mq5_code", "mqh_header", "log"]
    )

    # Build reasoning prompt from content
    reasoning_prompt = f"""
    {user_prompt}
    """

    if snippets_enable:
        reasoning_prompt += f"""
        Context:
        { "\n\n".join(all_texts) }
        """

    reasoning_prompt_token = count_tokens(reasoning_prompt, model_name, False)
    print(f"perform reasoning_prompt with total token of {reasoning_prompt_token} for analysis.")
    start = time.perf_counter()
    analysis = safe_invoke_ollama(reasoning_prompt, model_name, ollama_server)
    elapsed = time.perf_counter() - start
    print(f"analysis: {analysis} with {elapsed:.2f} seconds")

    return {
        "reasoning_prompt": reasoning_prompt,
        "analysis": analysis,
        "model_name": model_name,
        "run_id": run_id
    }

def store_analysis_snippets_json(reasoning_prompt, analysis, model_name=None):
    info = get_last_run_info()
    run_id = info["run_id"]
    archieve_dir = info["archieve"]   # consistent spelling

    if model_name is None:
        model_name = os.getenv("REASONING_AGENT")

    # Check archieve directory
    if not archieve_dir or not os.path.exists(archieve_dir):
        raise FileNotFoundError(f"❌ Archive directory not found: {archieve_dir}")

    archieve_file = os.path.join(archieve_dir, f"snippets_{run_id}.json")

    if not os.path.exists(archieve_file):
        raise FileNotFoundError(f"❌ Archive file not found for run_id {run_id}: {archieve_file}")

    # Load existing snippets JSON
    with open(archieve_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Add reasoning_prompt and analysis into the JSON
    if reasoning_prompt:
        data["reasoning_prompt"] = {
            "content": reasoning_prompt,
            "total_tokens": count_tokens(reasoning_prompt, model_name, False)
        }
    else:
        raise ValueError("❌ reasoning_prompt is None. Cannot store analysis.")

    if analysis:
        data["analysis"] = {
            "content": analysis,
            "total_tokens": count_tokens(analysis, model_name, False)
        }
    else:
        raise ValueError("❌ analysis is None. Cannot store analysis.")

    # Save back to the same file
    with open(archieve_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Updated snippets archieve with analysis and reasoning_prompt at {archieve_file}")

def parse_response_with_analysis(response: str):
    analysis_text = response
    old_code, fix_code, explanation = "", "", ""

    try:
        # Find the JSON block
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            json_str = match.group(0)
            parsed = json.loads(json_str)
            old_code = parsed.get("old_code", "")
            fix_code = parsed.get("fix_code", "")
            explanation = parsed.get("explanation", "")
            # Remove JSON from analysis text
            analysis_text = response.replace(json_str, "").strip()
    except Exception as e:
        print("JSON parse error:", e)

    return {
        "analysis_text": analysis_text,
        "old_code": old_code,
        "fix_code": fix_code,
        "explanation": explanation
    }

def suggest_code_improvements(
    ollama_server,
    last_reasoning_prompt,
    analysis,
    coder_prompt,
    fetch_snippets_enable=False
):
    print("start executing suggest_code_improvements ...")

    # --- Guard clause: check required inputs ---
    missing = []
    if not last_reasoning_prompt:
        missing.append("last_reasoning_prompt")
    if not analysis:
        missing.append("analysis")
    if not coder_prompt:
        missing.append("coder_prompt")

    data = {}
    archieve_dir = None
    run_id = None

    if missing:
        print("⚠️ Missing required inputs:", ", ".join(missing), ". Loading from archieve...")
        model_name = os.getenv("CODER_AGENT")
        data, archieve_dir, run_id = load_snippets(ollama_server, model_name)

        # Fill missing values from archieve JSON
        if not last_reasoning_prompt and "reasoning_prompt" in data:
            last_reasoning_prompt = data["reasoning_prompt"]["content"]
        if not analysis and "analysis" in data:
            analysis = data["analysis"]["content"]

    # --- Error if still missing ---
    if last_reasoning_prompt is None:
        raise ValueError("❌ reasoning_prompt is None. Cannot proceed.")
    if analysis is None:
        raise ValueError("❌ analysis is None. Cannot proceed.")
    if coder_prompt is None:
        raise ValueError("❌ coder_prompt is None. Cannot proceed.")

    snippets = []
    if fetch_snippets_enable:
        print("fetch full snippets for coder and header ...")
        start = time.perf_counter()
        code_snippets   = query_last_run_snippets(
            ollama_server, doc_type="code", section="mq5_code", top_k=100
        )
        header_snippets = query_last_run_snippets(
            ollama_server, doc_type="header", section="mqh_header", top_k=100
        )
        snippets = code_snippets + header_snippets
        elapsed = time.perf_counter() - start
        print(f"retrieved {len(snippets)} snippets in {elapsed:.2f} seconds")

    # Build coder prompt
    reasoning_coder_prompt = f"""Based on last reasoning prompt:\n{last_reasoning_prompt}\n
and based on last reasoning prompt analysis result:\n{analysis}\n
"""

    if fetch_snippets_enable:
        reasoning_coder_prompt += f"""
mql5 source code: {" ".join([s.page_content for s in code_snippets])}\n
header source code: {" ".join([s.page_content for s in header_snippets])}\n
"""

    reasoning_coder_prompt += f"""
{coder_prompt}\n
Return JSON with two keys: "old_code" and "fix_code".
old_code should show the original Trade_Strategy function.
fix_code should show the improved version.
    """

    print("generating code ...")
    start = time.perf_counter()
    model_name = os.getenv("CODER_AGENT")
    code_response = safe_invoke_ollama(reasoning_coder_prompt, model_name, ollama_server)
    elapsed = time.perf_counter() - start
    print(f"code_response: {code_response} with {elapsed:.2f} seconds")

    return parse_response_with_analysis(code_response)

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

def clean_log(input_file: str, archieve_folder: str = "archieve") -> str:
    """
    Remove the first 4 whitespace-separated columns from each line
    in the log file and write to a new file inside the archieve folder.
    Returns the path of the cleaned output file.
    """
    # Ensure archieve folder exists
    os.makedirs(archieve_folder, exist_ok=True)

    # Build output filename inside archieve folder
    base_name = os.path.basename(input_file)
    name, ext = os.path.splitext(base_name)
    output_file = os.path.join(archieve_folder, f"{name}_clean{ext}")

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

def record_git_commit_to_metadata_and_memory(run_id, commit_hash, repo_dir, archieve_folder):
    """
    Record the latest Git commit hash into metadata.json and memory.json.
    """
    # Update metadata.json
    metadata_path = os.path.join(archieve_folder, "metadata.json")
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
# run_id, archieve_folder, report_file, log_file = run_mt5_backtest(
#     ini_file,
#     TERMINAL_PATH,
#     BACKTEST_REPORT_PATH,
#     BACKTEST_LOG_PATH,
#     "logs",
#     False,
#     30,
# )
# print("run_id:", run_id)
# print("archieve_folder:", archieve_folder)
# print("report_file:", report_file)
# print("log_file:", log_file)

# json_report_file = report_tables_to_json(
#     report_file, 
#     archieve_folder=archieve_folder,
#     output_json="report_tables.json",
# )

# extract_Tester_report_summary(json_report_file)

# # generate vector id once stored to vector store
# vector_ids = process_run_for_embeddings(
#     ollama_server=OLLAMA_SERVER,
#     run_id=run_id,
#     archieve_folder=archieve_folder,
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
#     archieve_folder=archieve_folder,
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
# record_git_commit_to_metadata_and_memory(run_id, commit_hash, "MQL5", archieve_folder)




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
import chardet
from collections import defaultdict
from bs4 import BeautifulSoup

# Initialize tokenizer (cl100k_base works well for LLaMA‑style models)
enc = tiktoken.get_encoding("cl100k_base")

MAX_TOKENS = 32768
_ctx_cache = {}

ATTRIBUTE_GROUPS = {
    "attributes_timeframe": [
        "W_stage","diffMid_Trend","BBUpDn","trend","prev_trend","diffMid",
        "diffBBW","WLV","MidLV","UppLV","LowLV","close","high","low"
    ],
    "attributes_TRADEINFO": [
        "H2L_flyUP","H2L_flyDN","H2L_flyStrink","H2L_sideway",
        "L2H_flyUP","L2H_flyDN","L2H_flyStrink","L2H_sideway"
    ],
    "attributes_ORDERINFO": [
        "BUY_PROFIT","BUY_LOTS","SELL_PROFIT","SELL_LOTS","BUY_TICKET_NUM",
        "SELL_TICKET_NUM","BUYS","SELLS","TOTALORDERS"
    ],
    "attributes_ATRSL1buf": [
        "dir","Trend","LV","Upper","Lower","ATRSLMid","ATR_val"
    ],
    "attributes_BBTFImpact": [
        "HTF_Drive_LTF_Sideway","LTF_Drive_HTF_Fly","HTL_flyDN",
        "line_seq_touch","line_seq_cross","untouch_val","Midline_cross"
    ],
    "attributes_NEWORDEROPEN": [
        "TradeAct","OPEN_TICKET","OPEN_Type","OPEN_LOTS","OPEN_PRICE","OPEN_TIME",
        "CLOSED_TICKET","CLOSED_TYPE","CLOSED_LOT","CLOSED_PRICE","PROFIT","SWAP",
        "COMMISSION","FEE","TOTAL_PROFIT","TOTAL_SWAP","LAST_PROFIT"
    ],
    "attributes_NEWORDERCLOSE": [
        "TradeAct","OPEN_TICKET","OPEN_Type","OPEN_LOTS","DEAL_PRICE",
        "FREEMARGIN","MARGINREQUIRED"
    ]
}

timeframe_cats = {"M5","M15","M30","H1","H4","D1","W1"}

class StreamlitBrainstorming:
    def __init__(self):
        self.context = ""
        self.questions = []
        self.approaches = []
        self.design_sections = []
        self.current_step = 0
        
    def explore_context(self, project_path):
        # Analyze project files, recent commits, documentation
        pass
        
    def ask_question(self, question, options=None):
        # Present one question at a time
        st.write(f"**Question:** {question}")
        if options:
            return st.radio("Choose:", options)
        return st.text_input("Your response:")
        
    def propose_approaches(self, approaches):
        # Present 2-3 approaches with trade-offs
        for i, approach in enumerate(approaches):
            st.write(f"**Approach {i+1}: {approach['name']}")
            st.write(f"**Pros:** {approach['pros']}")
            st.write(f"**Cons:** {approach['cons']}")
            st.write(f"**Recommendation:** {approach['recommendation']}")
            
    def present_design(self, sections):
        # Present design sections
        for section in sections:
            st.subheader(section['title'])
            st.write(section['content'])

class JSONMemory:
    def __init__(self, path="memory.json"):
        self.path = path
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump({}, f)

    def store(self, key, value):
        data = self._load()
        data[key] = value
        self._save(data)

    def get(self, key):
        return self._load().get(key)

    def _load(self):
        with open(self.path, "r", encoding="utf-8") as f:
            content = f.read()
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON decode error: {e}. Attempting auto-clean.")
            content = content.replace(",}", "}").replace(",]", "]")
            return json.loads(content)

    def _save(self, data):
        try:
            json.dumps(data)  # validate serializable
        except Exception as e:
            raise ValueError(f"❌ Data not serializable: {e}")
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

def run_action(action, target, code_block=None):
    if action == "execute":
        if target.endswith(".sh"):
            result = subprocess.run(["bash", target], capture_output=True, text=True)
            return result.stdout
        elif target.endswith(".js") or target.endswith(".cjs"):
            result = subprocess.run(["node", target], capture_output=True, text=True)
            return result.stdout
        elif target.endswith(".py"):
            result = subprocess.run(["python", target], capture_output=True, text=True)
            return result.stdout
        elif code_block:
            cmd = code_block.split()[0]
            if cmd not in SAFE_COMMANDS:
                return f"Blocked unsafe command: {cmd}"
            result = subprocess.run(code_block, shell=True, capture_output=True, text=True)
            return result.stdout
    elif action == "include":
        with open(target, encoding="utf-8") as f:
            return f.read()
    return None

# --- Parse directives from LLM reply ---
def parse_directives(reply_text):
    actions = []
    for line in reply_text.splitlines():
        if line.startswith("execute:"):
            script = line.split(":",1)[1].strip()
            actions.append(("execute", script))
        elif line.startswith("include:"):
            file = line.split(":",1)[1].strip()
            actions.append(("include", file))
        elif line.startswith("ask_user:"):
            question = line.split(":",1)[1].strip()
            actions.append(("ask_user", question))
    return actions

def get_src_file_version(src_file: str):
    src_file = Path(src_file)
    with src_file.open("r", encoding="utf-8") as f:
        content = f.read()

    # Look for: #property version   "22.22"
    match = re.search(r'#property\s+version\s+"([\d\.]+)"', content)
    if match:
        return match.group(1)
    else:
        return None

def get_last_run_info():
    """Retrieve last run metadata from JSON-based memory."""
    ltm = JSONMemory(path="memory.json")
    return {
        "run_id": ltm.get("LAST_RUN_ID"),
        'run_id_num': ltm.get("LAST_RUN_ID_NUM"),
        'run_mq5_version': ltm.get("LAST_RUN_MQ5_VERSION"),
        "archieve": ltm.get("LAST_RUN_ARCHIEVE"),
        "rprev_un_id": ltm.get("PREV_LAST_RUN_ID"),
        'prev_run_id_num': ltm.get("PREV_LAST_RUN_ID_NUM"),
        "prev_archieve": ltm.get("PREV_LAST_RUN_ARCHIEVE"),
        "vector_db": {
            "code_embedding_id": ltm.get("LAST_RUN_CODE_EMBEDDING_ID"),
            "log_embedding_id": ltm.get("LAST_RUN_LOG_EMBEDDING_ID"),
            "report_embedding_id": ltm.get("LAST_RUN_REPORT_EMBEDDING_ID"),
            "header_embedding_id": ltm.get("LAST_RUN_HEADER_EMBEDDING_ID")  # singular for consistency
        }
    }

def beautify_text_area(raw_data):
    # 1. Handle empty or None data immediately
    if not raw_data:
        return ""

    # 2. Unpack the tuple if it exists
    if isinstance(raw_data, tuple):
        # Use first element if tuple isn't empty, else empty string
        filtered_data = raw_data[0] if len(raw_data) > 0 else ""
    else:
        # If it's already a string (or other type), use it as is
        filtered_data = raw_data

    # 3. Clean up all variants of escaped newlines
    # Ensure it's a string before calling replace
    clean_data = str(filtered_data).replace("\\\\n", "\n").replace("\\\n", "\n").replace("\\n", "\n")
    return clean_data

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

def get_model_ctx(model_name: str, base_url: str) -> int:
    if model_name in _ctx_cache:
        return _ctx_cache[model_name]
    url = f"{base_url}/api/show?model={model_name}"
    resp = requests.get(url)
    if resp.status_code == 200:
        data = resp.json()
        ctx_len = data.get("context_length", 8192)
        _ctx_cache[model_name] = ctx_len
        return ctx_len
    else:
        print(f"❌ Failed to query Ollama server: {resp.status_code}")
        return 8192

def safe_invoke_ollama(prompt: str, model_name: str, base_url: str, margin: float = 0.8):
    """
    Count tokens, chunk if needed, and sequentially invoke Ollama.
    margin = fraction of context length reserved for prompt (default 0.8 = 80%).
    """
    # max_tokens = get_model_ctx(model_name, base_url)
    if model_name == "qwen3-coder:30b-a3b-q4_K_M" or model_name == "qwen3.5:35b-a3b-q4_K_M":
        max_tokens = float(os.getenv("MAX_TOKENS"))
    else:
        max_tokens = float(os.getenv("EMBEDDED_AGENT_MAX_TOKENS"))
    safe_limit = int(max_tokens * margin)

    token_count = len(enc.encode(prompt))
    print(f"Prompt tokens: {token_count} / {max_tokens} (safe limit {safe_limit})")

    coder_model = OllamaLLM(model=model_name, base_url=base_url)

    if token_count <= safe_limit:
        return coder_model.invoke(prompt)
    else:
        print(f"Prompt exceeds safe limit {safe_limit}, chunking...")
        responses = []
        chunks = chunk_prompt(prompt)
        for idx, chunk in enumerate(chunks):
            start = time.perf_counter()
            print(f"Sending chunk {idx+1}/{len(chunks)} "
                  f"({len(enc.encode(chunk))} tokens)")
            resp = coder_model.invoke(chunk)
            elapsed = time.perf_counter() - start
            print(f"Elapsed for chunk {idx+1}: {elapsed:.2f} seconds")
            responses.append(resp)
        return "\n".join(responses)

def orchestrate_with_safe_invoke(prompt, skill_file, model_name, base_url):
    # Step 1: Load SKILL.md
    skill_doc = open(skill_file).read()
    context = skill_doc + "\n\nUser prompt:\n" + prompt

    # Step 2: Call Ollama safely
    llm_reply = safe_invoke_ollama(context, model_name, base_url)

    # Step 3: Parse directives from LLM reply
    actions = parse_directives(llm_reply)
    results = []
    for action, target in actions:
        if action == "ask_user":
            user_answer = st.text_input(f"LLM requests input: {target}")
            results.append(f"User answered: {user_answer}")
        else:
            output = run_action(action, target)
            results.append(f"{action} {target}:\n{output}")

    # Step 4: Feed results back into Ollama
    next_context = llm_reply + "\n\nResults:\n" + "\n".join(results)
    final_reply = safe_invoke_ollama(next_context, model_name, base_url)
    return final_reply

def get_log_attributes(log_file: str, target_minute: str = "01:05"):
    """
    Find the first detection of target_minute (ignoring date),
    capture the actual date, and return attributes for that block.
    Attributes are words before ':' or '['.
    Timeframe suffixes (_M5, _M15, _M30, _H1, _H4, _D1, _W1) are removed.
    """
    attributes_set = set()
    detected_time = None
    suffixes = ("_M5","_M15","_M30","_H1","_H4","_D1","_W1")

    for line in Path(log_file).read_text(encoding="utf-8").splitlines():
        # Match timestamp up to minutes
        ts_match = re.match(r"(\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}):\d{2}", line)
        if not ts_match:
            continue

        full_time = ts_match.group(1)  # e.g. '2025.03.03 01:05'
        minute_only = full_time.split(" ")[1]  # e.g. '01:05'

        if minute_only == target_minute:
            if detected_time is None:
                detected_time = full_time  # lock onto first detection
            if line.startswith(detected_time):
                # Attributes before ":" or before "["
                for kv in re.findall(r"(\w+)(?=:|\[)", line):
                    for suf in suffixes:
                        if kv.endswith(suf):
                            kv = kv[:-len(suf)]
                    # skip numeric-only attributes
                    if kv.isdigit():
                        continue
                    attributes_set.add(kv)
        elif detected_time:
            # stop scanning once we move past the target minute block
            break

    return sorted(attributes_set)

def clean_attribute(key: str) -> str:
    """Remove timeframe suffix (_M5, _M15, etc.) from attribute names."""
    return re.sub(r'_(M5|M15|M30|H1|H4|D1|W1)$', '', key)

def try_cast(x: str):
    """Convert to float if possible, else return string."""
    x = x.lstrip("[")  # strip stray leading “[”
    try:
        return float(x)
    except ValueError:
        return x

def parse_alltf(val: str):
    result = {}
    val = val.strip().lstrip("[").rstrip("]")

    tokens = [t.strip() for t in val.split(',') if t.strip()]
    for tok in tokens:
        m = re.match(r'([A-Z0-9]+)_(.+)', tok)
        if m:
            tf, num = m.groups()
            num = num.split(',')[0].strip()
            result[tf] = num
    return result

def parse_value(val: str, keep_array=False):
    """
    Extract values from arrays.
    - If keep_array=True (for AllTF), return full array.
    - Otherwise, return only the first numeric/string value.
    - Remove leading keyword in parentheses and stray “[”.
    """
    val = re.sub(r'^\([^)]*\)', '', val).strip()

    arr_match = re.search(r'\[([^\]]+)\]', val)
    if arr_match:
        nums = [n.strip().lstrip("[") for n in arr_match.group(1).split(',') if n.strip()]
        if keep_array:
            return [try_cast(n) for n in nums]
        else:
            return try_cast(nums[0]) if nums else None
    else:
        return try_cast(val)

def truncate_to_minute(timestamp: str) -> str:
    """Drop seconds from timestamp, keep only YYYY.MM.DD HH:MM."""
    return timestamp[:-3]

def smart_cast(val: str):
    """Try to cast to float/int, else return string."""
    v = val.strip()
    try:
        if "." in v:
            return float(v)
        return int(v)
    except ValueError:
        return val.strip()

def build_nested_structure(log_file: str, archive_folder: str):
    data = defaultdict(lambda: defaultdict(dict))

    for line in Path(log_file).read_text(encoding="utf-8").splitlines():
        ts_match = re.match(r"(\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2})", line)
        if not ts_match:
            continue
        timestamp = truncate_to_minute(ts_match.group(1))

        tf_match = re.search(r"\[(.*?)\]", line)
        if not tf_match:
            continue
        timeframe = tf_match.group(1)

         # --- ORDERINFO branch ---
        # --- Flat key:value categories ---
        if timeframe in ("ORDERINFO", "NEW_ORDER_OPEN", "NEW_ORDER_CLOSE"):
            for raw_key, raw_val in re.findall(r"(\w+):([^,]+)", line):
            # for raw_key, raw_val in re.findall(r"(\w+):([^\s]+)", line):
                attr_name = clean_attribute(raw_key)
                if attr_name.isdigit():
                    continue
                # try numeric cast, else keep string
                try:
                    val = float(raw_val) if "." in raw_val else int(raw_val)
                except ValueError:
                    val = raw_val.strip()
                data[timestamp][timeframe][attr_name] = val
            continue
        # --- TRADEINFO branch ---
        elif timeframe == "TRADEINFO":
            # store TRADEINFO under a special key as a list
            if "TRADEINFO" not in data[timestamp]:
                data[timestamp]["TRADEINFO"] = []
            gate_blocks = re.findall(r"Gate:\[([^\]]+)\]([^G]+)", line)
            for gate_name, attrs in gate_blocks:
                gate_info = {"Gate": gate_name.strip()}
                for kv in attrs.strip().split():
                    if ":" in kv:
                        k, v = kv.split(":", 1)
                        gate_info[k] = smart_cast(v)
                data[timestamp]["TRADEINFO"].append(gate_info)
            continue
        # --- BBTFImpact / BB_data branch ---
        for raw_key, raw_val in re.findall(r"(\w+):(\[.*?\])", line):
            attr_name = clean_attribute(raw_key)

            if attr_name.isdigit():
                continue

            if timeframe == "BBTFImpact":
                # raw_val here is the whole "[M5_8-7,8, M15_0-9,9, ...]"
                val = parse_alltf(raw_val)
            else:
                val = parse_value(raw_val)

            if val is not None:
                data[timestamp][timeframe][attr_name] = val

    archive_path = Path(archive_folder)
    archive_path.mkdir(parents=True, exist_ok=True)
    json_file = archive_path / "filtered_log_json.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Cleaned JSON stored at {json_file}")
    return json_file

def reorder_timeframe_attrs(attrs):
    reordered = []
    # If first_stage exists, put it first
    if "first_stage" in attrs:
        reordered.append("first_stage")
    # If W_stage exists, ensure it's second
    if "W_stage" in attrs:
        # If first_stage not present, insert synthetic first_stage first
        if "first_stage" not in attrs:
            reordered.append("first_stage")  # synthetic placeholder
        reordered.append("W_stage")
    # Add the rest in natural order, skipping duplicates
    for a in attrs:
        if a not in ("first_stage","W_stage"):
            reordered.append(a)
    return reordered

def json_log_to_excel(json_file: str, archive_folder: str, base_name: str = "log_matrix"):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    category_attrs = {}

    # --- Pass 1: build unified headers ---
    for ts, categories in data.items():
        for cat, content in categories.items():
            if isinstance(content, dict):
                attr_names = list(content.keys())
                if cat in timeframe_cats:
                    attr_names = reorder_timeframe_attrs(attr_names)
                category_attrs[cat] = attr_names

            elif cat == "TRADEINFO" and isinstance(content, list):
                for idx, entry in enumerate(content):
                    if isinstance(entry, dict):
                        keys = []
                        for k in entry.keys():
                            if k == "cnt":
                                keys.append("Trade_act")
                            else:
                                keys.append(k)
                        category_attrs[f"TRADEINFO{idx}"] = keys

    # --- Build merged headers without duplicates ---
    max_cols = max(len(v) for v in category_attrs.values())
    headers = []
    seen = set()
    for i in range(max_cols):
        parts = []
        for cat, attrs in category_attrs.items():
            if i < len(attrs):
                attr = attrs[i]
                if attr in seen:
                    continue
                parts.append(attr)
                seen.add(attr)
        if parts:
            headers.append("/".join(parts))
    print("header:", headers)

    # --- Pass 2: Collect values aligned to headers ---
    rows = []
    for ts, categories in data.items():
        for cat, content in categories.items():
            if isinstance(content, dict):
                attr_dict = {}
                for k,v in content.items():
                    if isinstance(v, dict):
                        items = [f"{subk}={subv}" for subk, subv in v.items()]
                        # attr_dict[k] = items if items else []
                        attr_dict[k] = "{" + "|".join(items) + "}"
                    else:
                        attr_dict[k] = v   # don’t wrap in str()

                row = {"datetime": ts, "category": cat}
                for i,h in enumerate(headers, start=1):
                    attr_names = h.split("/")
                    val = ""
                    for name in attr_names:
                        if name in attr_dict:
                            val = attr_dict[name]
                            break
                    row[f"Col{i}"] = val
                rows.append(row)

            elif cat == "TRADEINFO" and isinstance(content, list):
                for idx, entry in enumerate(content):
                    attr_dict = {}
                    for k,v in entry.items():
                        if k == "cnt":
                            attr_dict["Trade_act"] = str(v)
                        else:
                            attr_dict[k] = str(v)

                    row = {"datetime": ts, "category": f"TRADEINFO{idx}"}
                    for i,h in enumerate(headers, start=1):
                        attr_names = h.split("/")
                        val = ""
                        for name in attr_names:
                            if name in attr_dict:
                                val = attr_dict[name]
                                break
                        row[f"Col{i}"] = val
                    rows.append(row)

   # --- Build DataFrame ---
    df = pd.DataFrame(rows)

    # ensure all ColN exist
    for i in range(1, len(headers)+1):
        colname = f"Col{i}"
        if colname not in df.columns:
            df[colname] = ""

    ordered_cols = ["datetime","category"]+[f"Col{i}" for i in range(1,len(headers)+1)]
    df = df[ordered_cols]

    # --- Simplify merged headers using category_attrs ---
    headers_simple = []
    seen = set()
    for i in range(max(len(v) for v in category_attrs.values())):
        parts = []
        for cat, attrs in category_attrs.items():
            if i < len(attrs):
                attr = attrs[i]
                # collapse timeframe categories
                if cat in timeframe_cats:
                    key = f"TF-{attr}"
                # collapse TRADEINFO categories
                # elif cat.startswith("TRADEINFO"):
                #     key = f"TRADEINFO-{attr}"
                else:
                    key = f"{cat}-{attr}"
                if key not in seen:
                    parts.append(key)
                    seen.add(key)
        if parts:
            headers_simple.append("/".join(parts))

    # rename ColN to simplified headers
    df.columns = ["datetime","category"]+headers_simple

    print("headers_simple:", headers_simple)

    # Save
    archive_path = Path(archive_folder)
    archive_path.mkdir(parents=True, exist_ok=True)

    csv_path = archive_path / f"{base_name}.csv"
    xlsx_path = archive_path / f"{base_name}.xlsx"

    df.to_csv(csv_path, index=False, encoding="utf-8")
    df.to_excel(xlsx_path, index=False, engine="openpyxl")

    return df, csv_path, xlsx_path

def build_dataframe_from_log(log_file: str, selected_attributes: list, archieve_folder: str):
    """
    Parse a log file, build a DataFrame, and filter to selected attributes.
    """
    rows = []
    for line in Path(log_file).read_text(encoding="utf-8").splitlines():
        ts_match = re.match(r"(\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2})", line)
        if not ts_match:
            continue
        timestamp = ts_match.group(1)

        # extract category
        cat_match = re.search(r"\[(.*?)\]", line)
        category = cat_match.group(1) if cat_match else "Unknown"

        # extract metrics
        metrics = {}
        for kv in re.findall(r"(\w+):([-\d\.]+)", line):
            key, val = kv
            if key in selected_attributes:  # only keep chosen attributes
                metrics[key] = float(val)

        rows.append({"datetime": timestamp, "timeframe": category, **metrics})

    df = pd.DataFrame(rows)
    # --- Save to archive folder ---
    archive_path = Path(archieve_folder)
    archive_path.mkdir(parents=True, exist_ok=True)  # ensure folder exists

    # Save as CSV
    csv_file = archive_path / "filtered_log.csv"
    df.to_csv(csv_file, index=False, encoding="utf-8")
    print(f"✅ DataFrame stored as CSV at {csv_file}")

    # # Save as Excel (optional)
    # excel_file = archive_path / "filtered_log.xlsx"
    # df.to_excel(excel_file, index=False, engine="openpyxl")
    # print(f"✅ DataFrame stored as Excel at {excel_file}")

    return df

def copyfiles(
    code_repo: str,
    mq5_file: str,
    header_file: str,
    mql5_path: str,
    mq5_selected: bool,
    header_selected: bool,
    trade_header_selected: bool,
    backtested_data_selected: bool,
    copy_to_from: bool,
):
    """
    Copy EA (.mq5) and header (.mqh) files from the code repo into the MQL5 directory.
    If src/dst is a directory, copy the entire directory tree.
    Only copies files if their *_selected flag is True.
    Ensures destination directories exist and prints status messages.
    Returns the latest commit message (or SHA) from the repo if copy succeeds,
    otherwise returns None.
    """
    info = get_last_run_info()
    archive_dir = info.get("archive") or info.get("archieve")
    run_mq5_version = info.get("run_mq5_version")

    if not archive_dir:
        print("❌ Archive directory not found in get_last_run_info() result")
        return {}, archive_dir, run_id, run_id_num

    project_trade_header_file = os.path.join(r"C:\Users\Tofy3\Project\bb_mtf_strategy", os.getenv("HEADRER_SCRIPT_SUBPATH"))
    coderepo_trade_header_file = os.path.join(code_repo, os.getenv("TRADE_HEADER_SUBPATH"))
    mt5_trade_header_file = os.path.join(mql5_path, os.getenv("TRADE_HEADER_SUBPATH"))
    project_backtested_data_path = os.path.join(r"C:\Users\Tofy3\Project\bb_mtf_strategy", os.path.join(r"references\Backtest_data", run_mq5_version))

    if copy_to_from: 
        copy_files_path = [ # files_to_MT5_copy
            (mq5_selected, os.path.join(code_repo, mq5_file), os.path.join(mql5_path, mq5_file)),
            (header_selected, os.path.join(code_repo, header_file), os.path.join(mql5_path, header_file)),
            (trade_header_selected, coderepo_trade_header_file, mt5_trade_header_file),
            (backtested_data_selected, archive_dir, project_backtested_data_path),
        ]
    else:
        copy_files_path = [ # files_from_project_copy
            (trade_header_selected, project_trade_header_file, coderepo_trade_header_file)
        ]

    success = True

    if copy_files_path:
        for selected, src, dst in copy_files_path:
            if not selected:
                print(f"⏭️ Skipped {src} (not selected)")
                continue

            if os.path.isfile(src):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                try:
                    shutil.copy(src, dst)
                    print(f"✅ Copied file {src} → {dst}")
                except Exception as e:
                    print(f"❌ Failed to copy file {src} → {dst}: {e}")
                    success = False

            elif os.path.isdir(src):
                try:
                    # Copy entire directory tree
                    if os.path.exists(dst):
                        shutil.rmtree(dst)  # remove old copy
                    shutil.copytree(src, dst)
                    print(f"📂 Copied directory {src} → {dst}")
                except Exception as e:
                    print(f"❌ Failed to copy directory {src} → {dst}: {e}")
                    success = False

            else:
                print(f"❌ Source not found: {src}")
                success = False

        if success:
            try:
                result = subprocess.run(
                    ["git", "log", "-1", "--pretty=%H %s"],
                    cwd=code_repo,
                    capture_output=True,
                    text=True,
                    check=True
                )
                latest_commit = result.stdout.strip()
                return latest_commit
            except subprocess.CalledProcessError as e:
                print(f"❌ Could not retrieve latest commit: {e}")
                return None
        else:
            return None


def extract_errors(compile_output: str) -> list[str]:
    """
    Extract only error lines from compiler output.
    Returns a list of error messages.
    """
    errors = []
    for line in compile_output.splitlines():
        # Match lines containing 'error' but not 'warning'
        if "error" in line.lower() and "warning" not in line.lower():
            errors.append(line.strip())
    return errors

def compile_ea(mq5_file: str, metaeditor_path: str):
    """
    Compile an EA and return the log content.
    """
    print(f"Start compile EA: {mq5_file}...")
    mq5_path = Path(mq5_file).resolve()
    
    # MetaEditor creates the log file in the same directory as the source file
    # Example: Tofu_EA_Simple.mq5 -> Tofu_EA_Simple.log
    expected_log_file = mq5_path.with_suffix('.log')
    
    # Remove old log if it exists to avoid reading old results
    if expected_log_file.exists():
        expected_log_file.unlink()

    cmd = [
        metaeditor_path,
        f"/compile:{mq5_path}",
        "/log"
    ]
    
    # Use 'shell=False' (default) and wait for the process to finish
    subprocess.run(cmd, capture_output=True, text=True)
    
    print("Done compile EA.")

    # Read the log file (MetaEditor uses UTF-16 encoding)
    if expected_log_file.exists():
        try:
            log_content = expected_log_file.read_text(encoding="utf-16")
            return log_content
        except Exception as e:
            return f"Error reading log file: {e}"
    
    return "Log file not found. Compilation might have failed to start."

def compile_fail_update_memory(base_path=""):
    """
    On compile fail, update memory.json by incrementing LAST_RUN_ID_NUM.
    """
    memory_file = Path(base_path) / "memory.json"

    if not memory_file.exists():
        raise FileNotFoundError(f"❌ memory.json not found at {memory_file}")

    try:
        # Load memory.json
        with memory_file.open("r", encoding="utf-8") as f:
            memory_data = json.load(f)

        # Increment LAST_RUN_ID_NUM
        current_num = memory_data.get("LAST_RUN_ID_NUM", 0)
        memory_data["LAST_RUN_ID_NUM"] = current_num + 1

        # Save back
        with memory_file.open("w", encoding="utf-8") as f:
            json.dump(memory_data, f, indent=2)

        print(f"✅ Updated LAST_RUN_ID_NUM to {memory_data['LAST_RUN_ID_NUM']} in {memory_file}")
        return memory_data["LAST_RUN_ID_NUM"]

    except Exception as e:
        print(f"❌ Failed to update memory.json: {e}")
        return None

# def update_ini_file(
#     ini_path,
#     login,
#     password,
#     server,
#     expert,
#     symbol="XAUUSD",
#     period="M5",
#     from_date="2025.03.01",
#     to_date="2025.04.01",
#     deposit=10000,
#     currency="USD",
#     leverage="1:100",
#     visual=False,
#     report_path=r"C:\Users\Tofy3\Downloads\Tester_report.html",
# ):
#     tester_updates = {
#         "Expert": expert,
#         "Symbol": symbol,
#         "Period": period,
#         "Optimization": "0",
#         "Model": "0",
#         "FromDate": from_date,
#         "ToDate": to_date,
#         "ForwardMode": "0",
#         "Deposit": str(deposit),
#         "Currency": currency,
#         "ProfitInPips": "0",
#         "Leverage": leverage,
#         "ExecutionMode": "0",
#         "OptimizationCriterion": "0",
#         "Visual": int(visual),
#         "ShutdownTerminal": "1",
#         "ReplaceReport": "1",
#         "Report": report_path,   # ✅ ensure Report is always present
#     }

#     with open(ini_path, "r", encoding="utf-16") as f:
#         lines = f.readlines()

#     new_lines = []
#     in_tester = False
#     seen_keys = set()
#     tester_end_index = None

#     for line in lines:
#         stripped = line.strip()

#         # Detect section headers
#         if stripped.startswith("[") and stripped.endswith("]"):
#             if stripped.lower() == "[tester]":
#                 in_tester = True
#             else:
#                 if in_tester and tester_end_index is None:
#                     tester_end_index = len(new_lines)  # mark end of Tester section
#                 in_tester = False
#             new_lines.append(line)
#             continue

#         if in_tester and "=" in stripped:
#             key = stripped.split("=", 1)[0]
#             if key in tester_updates:
#                 new_lines.append(f"{key}={tester_updates[key]}\n")
#                 seen_keys.add(key)
#                 print(f"Updated attribute: {key}={tester_updates[key]}")
#             else:
#                 new_lines.append(line)
#         else:
#             new_lines.append(line)

#     # If Tester section ended before TesterInputs, insert missing keys there
#     if tester_end_index is not None:
#         missing = [f"{k}={v}\n" for k, v in tester_updates.items() if k not in seen_keys]
#         if missing:
#             print("New attributes added to [Tester]:")
#             for m in missing:
#                 print("  " + m.strip())
#         new_lines = new_lines[:tester_end_index] + missing + new_lines[tester_end_index:]

#     # with open(ini_path, "w") as f:
#     #     f.writelines(new_lines)
#     with open(ini_path, "w", encoding="utf-16", newline="\r\n") as f:
#         f.writelines(new_lines)

#     print("Updated ini file:", ini_path)
#     return str(ini_path)

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
    visual=False,
    report_path=r"C:\Users\Tofy3\Downloads\Tester_report.html",
    version_file=None,   # <-- pass in your MQH file path
):
    # get version from MQH
    version = get_src_file_version(os.path.normpath(version_file))
    if version:
        ord_comment = f"V{version}"
    else:
        ord_comment = "VUNKNOWN"

    tester_updates = {
        "Expert": expert,
        "Symbol": symbol,
        "Period": period,
        "Optimization": "0",
        "Model": "0",
        "FromDate": from_date,
        "ToDate": to_date,
        "ForwardMode": "0",
        "Deposit": str(deposit),
        "Currency": currency,
        "ProfitInPips": "0",
        "Leverage": leverage,
        "ExecutionMode": "263",
        "OptimizationCriterion": "0",
        "Visual": int(visual),
        "ShutdownTerminal": "1",
        "ReplaceReport": "1",
        "Report": report_path,
    }

    with open(ini_path, "r", encoding="utf-16") as f:
        lines = f.readlines()

    new_lines = []
    in_tester = False
    in_inputs = False
    seen_keys = set()

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("[") and stripped.endswith("]"):
            in_tester = stripped.lower() == "[tester]"
            in_inputs = stripped.lower() == "[testerinputs]"
            new_lines.append(line)
            continue

        if in_tester and "=" in stripped:
            key = stripped.split("=", 1)[0]
            if key in tester_updates:
                new_lines.append(f"{key}={tester_updates[key]}\n")
                seen_keys.add(key)
            else:
                new_lines.append(line)
        elif in_inputs and stripped.startswith("ORDERS_COMMENT="):
            # overwrite ORDERS_COMMENT with MQH version
            new_lines.append(f"ORDERS_COMMENT={ord_comment}\n")
            print(f"Updated ORDERS_COMMENT={ord_comment}")
        else:
            new_lines.append(line)

    with open(ini_path, "w", encoding="utf-16", newline="\r\n") as f:
        f.writelines(new_lines)

    print("Updated ini file:", ini_path)
    return str(ini_path)

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
    with open(config_path, "r", encoding="utf-16") as f:
        for line in f:
            if line.strip().lower().startswith("report="):
                report_file = line.strip().split("=", 1)[1].strip()
                break

    if report_file and not os.path.isabs(report_file):
        report_file = os.path.join(report_path, report_file)
        print(f"get full report_file:{report_file}")

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
            print(f"copied latest_log_file:{latest_log_file} to archieve_folder:{archieve_folder}")
    print(f"done backtesting with run_id:{run_id}, archieve_folder:{archieve_folder}, report_file:{report_file}, latest_log_file:{latest_log_file}")
    return run_id, archieve_folder, report_file, latest_log_file

def mt5_html_to_xlsx(html_path: str):
    # --- Build XLSX path from HTML path ---
    base, _ = os.path.splitext(html_path)
    xlsx_path = base + ".xlsx"

    # --- Load HTML ---
    with open(html_path, "r", encoding="utf-16") as f:  # MT5 often uses UTF-16
        soup = BeautifulSoup(f, "html.parser")

    # --- Extract tables ---
    tables = pd.read_html(str(soup))

    # --- Write to XLSX ---
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for idx, df in enumerate(tables):
            sheet_name = f"Table_{idx}"
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Saved XLSX: {xlsx_path}")
    return xlsx_path

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

def compress_table(table):
    compressed = {}
    for item in table:
        if "label" in item and "value" in item:
            # label/value pair → key:value
            key = item["label"].rstrip(":")
            compressed[key] = item["value"]
        elif "value" in item:
            val = item["value"]
            if "=" in val:
                key, value = val.split("=", 1)
                compressed[key] = value
            else:
                # fallback if no '=' and no label
                compressed[val] = None
    return compressed

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

    compressed_data = {}
    for key, table in raw_data.items():
        if key in ("table_0", "table_inputs", "table_results"):
            compressed_data[key] = compress_table(table)
        else:
            compressed_data[key] = table

    # Save cleaned report
    report_tables_clean_file = json_file.parent / "report_tables_clean.json"
    with report_tables_clean_file.open("w", encoding="utf-8") as f:
        json.dump(compressed_data, f, indent=4, ensure_ascii=False)
    print(f"Cleaned report saved to: {report_tables_clean_file}")

    # Remove unnecessary tables if present
    for key in ("table_0", "table_inputs"):
        if key in compressed_data:
            del compressed_data[key]

    # Delete original file safely
    if json_file.exists():
        json_file.unlink()
        print(f"Original file removed: {json_file}")

    return compressed_data, report_tables_clean_file

def load_params_from_ini(ini_file):
    """
    Load parameters from an INI file into a dict.
    Optionally update ORDERS_COMMENT to a new value.
    """
    config = configparser.ConfigParser()
    # config.read(ini_file)
    with open(ini_file, "r", encoding="utf-16") as f:
        config.read_file(f)

    # Assume parameters are under a section called [Parameters]
    params = dict(config["Tester"])
    params["ORDERS_COMMENT"] = config["TesterInputs"].get("ORDERS_COMMENT")
    return params

def save_run_and_update_memory(run_id, parameters, archieve_folder, report_files, log_file, vector_ids):
    
    # extract ORDER_COMMENT from report_file
    print("Type of report_files:", type(report_files))
    # report_files_path = os.path.join(archieve_folder, report_files)
    report_files_path = os.path.join(archieve_folder, report_files[0])
    print("report_files_path:", report_files_path)
    if not os.path.isfile(report_files_path):
        print(f"❌ File not found: {report_files_path}")
        return None
    with open(report_files_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ORDERS_COMMENT = data.get("table_inputs", {}).get("ORDERS_COMMENT")

    metadata = {
        "run_id": run_id,
        "run_id_num": 0,
        "run_mq5_version": ORDERS_COMMENT,
        "timestamp": datetime.now().isoformat(),
        "parameters": parameters,
        "artifacts": {
            "archieve_folder": archieve_folder,
            "report_variants": report_files,
            "log_file": log_file,
        },
        "vector_db": vector_ids,
    }
    

    # --- Save metadata.json inside archive folder ---
    os.makedirs(archieve_folder, exist_ok=True)
    metadata_path = Path(archieve_folder) / "metadata.json"
    metadata_num = 0

    metadata_path = os.path.join(archieve_folder, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    # --- Update memory.json ---
    ltm = JSONMemory(path="memory.json")
    prev_run_id = ltm.get("LAST_RUN_ID")
    prev_run_id_num = ltm.get("LAST_RUN_ID_NUM")
    prev_run_mq5_version = ltm.get("LAST_RUN_MQ5_VERSION")
    prev_archieve = ltm.get("LAST_RUN_ARCHIEVE")

    ltm.store("LAST_RUN_ID", run_id)
    ltm.store("LAST_RUN_ID_NUM", 0)
    ltm.store("LAST_RUN_MQ5_VERSION", ORDERS_COMMENT)
    ltm.store("LAST_RUN_ARCHIEVE", archieve_folder)

    ltm.store("LAST_RUN_CODE_EMBEDDING_ID", vector_ids.get("code_embedding_id"))
    ltm.store("LAST_RUN_HEADER_EMBEDDING_ID", vector_ids.get("header_embedding_id"))
    ltm.store("LAST_RUN_LOG_EMBEDDING_ID", vector_ids.get("log_embedding_id"))
    ltm.store("LAST_RUN_REPORT_EMBEDDING_ID", vector_ids.get("report_embedding_id"))
    if prev_run_id and prev_run_id_num and prev_archieve:
        if prev_run_id != run_id or prev_run_id_num != 0:
            ltm.store("PREV_LAST_RUN_ID", prev_run_id)
            ltm.store("PREV_LAST_RUN_ID_NUM", prev_run_id_num)
            ltm.store("LAST_RUN_MQ5_VERSION", prev_run_mq5_version)
            ltm.store("PREV_LAST_RUN_ARCHIEVE", prev_archieve)
            print(f"updated PREV_LAST_RUN")

    # Insert newest run at the top of RUN_HISTORY list
    history = ltm.get("RUN_HISTORY") or []
    history.insert(0, metadata)
    ltm.store("RUN_HISTORY", history)

    print(f"✅ Saved run {run_id} with num {metadata_num} and updated memory.json")

def read_file_safely(file_path):
    ext = Path(file_path).suffix.lower()
    with open(file_path, "rb") as f:
        raw_bytes = f.read()

    if ext == ".mq5":
        # mq5 files are typically UTF-16LE
        try:
            text = raw_bytes.decode("utf-16", errors="replace")
        except UnicodeError:
            text = raw_bytes.decode("utf-8", errors="replace")
    else:
        # headers, logs, txt are usually UTF-8
        text = raw_bytes.decode("utf-8", errors="replace")

    return text.replace("\x00", "")

def embed_and_store(ollama_server, archive_folder, file_path, vector_db_path, doc_type, run_id,
                    faiss_embedded=True, chunk_size=2000, chunk_overlap=200):
    # --- Read file safely depending on extension ---
    ext = Path(file_path).suffix.lower()
    with open(file_path, "rb") as f:
        raw_bytes = f.read()

    if ext == ".mq5":
        # mq5 files are typically UTF-16LE
        try:
            text = raw_bytes.decode("utf-16", errors="replace")
        except UnicodeError:
            text = raw_bytes.decode("utf-8", errors="replace")
    else:
        # headers, logs, txt are usually UTF-8
        text = raw_bytes.decode("utf-8", errors="replace")

    text = text.replace("\x00", "")  # remove nulls

    file_path_str = str(file_path)

    # --- Choose splitter ---
    if file_path_str.endswith((".mq5", ".mqh")):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ";", "}", "{"]
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    docs = splitter.create_documents([text])

    # --- Section tagging + grouping ---
    section_map = {}
    for d in docs:
        section = "general"
        if "table_deals" in d.page_content:
            section = "table_deals"
        elif "table_orders" in d.page_content:
            section = "table_orders"
        elif "table_results" in d.page_content:
            section = "table_results"
        elif file_path_str.endswith(".mq5"):
            section = "mq5_code"
        elif file_path_str.endswith(".mqh"):
            section = "mqh_header"
        elif file_path_str.endswith(".log") or (file_path_str.endswith(".txt") and "log_part" in file_path_str):
            section = "log"

        if section not in section_map:
            section_map[section] = {
                "file": file_path_str,
                "chunks": [],
                "token": 0
            }
        section_map[section]["chunks"].append(d.page_content)
        section_map[section]["token"] += len(d.page_content.split())

    # --- Archive path ---
    os.makedirs(archive_folder, exist_ok=True)
    archive_path = Path(archive_folder) / f"snippets_{run_id}.json"

    info = get_last_run_info()
    run_id_num = info["run_id_num"]

    if archive_path.exists():
        try:
            archive = json.loads(archive_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            archive = {"raw_data": {}}
    else:
        archive = {"raw_data": {}}

    # Ensure structure
    archive.setdefault("raw_data", {})
    archive["raw_data"].setdefault(run_id, {})

    # Merge sections instead of overwriting
    for section, data in section_map.items():
        if section not in archive["raw_data"][run_id]:
            archive["raw_data"][run_id][section] = data
        else:
            # append chunks and update token count
            archive["raw_data"][run_id][section]["chunks"].extend(data["chunks"])
            archive["raw_data"][run_id][section]["token"] += data["token"]

    archive_path.write_text(json.dumps(archive, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Updated JSON archive at {archive_path}")

    # --- Embed and store in FAISS ---
    if faiss_embedded:
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
        print(f"✅ Updated FAISS vector store at {vector_db_path}")

    return f"{doc_type}_{run_id}"

def process_run_for_embeddings(
    ollama_server, archieve_folder, run_id,
    ea_source_file, log_file, report_file, 
    header_files=None, faiss_embedded=True,
    log_chunk_size=10000   # number of lines per chunk
):
    vector_db_path = "files_index"

    def store_with_stats(file_path, doc_type, archieve_folder, faiss_embedded=True):
        if file_path and os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"storing {doc_type} = {file_path} (size: {size_mb:.2f} MB)...")
            start = time.perf_counter()
            embedding_id = embed_and_store(ollama_server, archieve_folder, file_path, vector_db_path, doc_type, run_id, faiss_embedded)
            elapsed = time.perf_counter() - start
            print(f"{doc_type} stored in {elapsed:.2f} seconds (embedding_id={embedding_id})")
            return embedding_id
        else:
            print(f"{doc_type} file not found, skipping...")
            return None

    print(f"Perform storing of {run_id} code with faiss={faiss_embedded} ...")
    # EA source file
    code_id = store_with_stats(ea_source_file, "code", archieve_folder, faiss_embedded)
    print(f"Perform storing of {run_id} header with faiss={faiss_embedded} ...")
    # Header file(s) – if you want multiple headers, loop here
    header_id = store_with_stats(header_files, "header", archieve_folder, faiss_embedded)
    print(f"Perform storing of {run_id} log with faiss={faiss_embedded} ...")
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
            chunk_id = embed_and_store(ollama_server, archieve_folder, chunk_file, vector_db_path, "log", run_id, faiss_embedded)
            elapsed = time.perf_counter() - start
            print(f"log chunk {i//log_chunk_size} stored in {elapsed:.2f} seconds")
            chunk_ids.append(chunk_id)

        # unify into a single identifier
        log_id = f"log_{run_id}"
    else:
        print("log_file not found, skipping...")

    print(f"Perform storing of {run_id} report with faiss={faiss_embedded} ...")
    # Report file
    report_id = store_with_stats(report_file, "report", archieve_folder, faiss_embedded)

    print("done storing to vector at", vector_db_path)
    return {
        "code_embedding_id": code_id,
        "header_embedding_id": header_id,
        "log_embedding_id": log_id,
        "report_embedding_id": report_id
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

    # ✅ Wrap FAISS as retriever
    retriever = db.as_retriever(search_kwargs={"k": top_k})

    # If query_text provided, do semantic search
    if query_text:
        results = retriever.get_relevant_documents(query_text)
    else:
        # fallback: return all docs
        results = list(db.docstore._dict.values())

    # ✅ Apply metadata filters
    filtered = [r for r in results if r.metadata.get("run_id") == run_id]

    if doc_type:
        filtered = [r for r in filtered if r.metadata.get("doc_type") == doc_type]

    if section:
        filtered = [r for r in filtered if r.metadata.get("section") == section]

    print(f"filtered len:{len(filtered)}, current top_k:{top_k}")
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
    code_snippets, archieve_dir, run_id = query_last_run_snippets(ollama_server, "code", "mq5_code", 100)
    header_snippets, archieve_dir, run_id = query_last_run_snippets(ollama_server, "header", "mqh_header", 100)
    log_snippets, archieve_dir, run_id = query_last_run_snippets(ollama_server, "log", "log", 70)

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

    # Archive file path
    archieve_file = os.path.join(archieve_dir, f"snippets_{run_id}.json")

    info = get_last_run_info()
    run_id_num = info["run_id_num"]

    if os.path.exists(archieve_file):
        archive = json.loads(Path(archieve_file).read_text(encoding="utf-8"))
    else:
        archive = {"raw_data": {run_id: {}}}

    if "raw_data" not in archive:
        archive["raw_data"] = {}
    if run_id not in archive["raw_data"]:
        archive["raw_data"][run_id] = {}
    # if run_id_num not in archive["raw_data"][run_id]:
        # archive["raw_data"][run_id][run_id_num] = {}

    # Add snippets JSON
    snippets_json = {"snippets": json_snippets, "total_tokens": total_tokens}
    # archive["raw_data"][run_id][run_id_num][section] = snippets_json
    archive["raw_data"][run_id][section] = snippets_json
    # Save JSON
    Path(archieve_file).write_text(json.dumps(archive, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Stored latest snippets with id {run_id} to {archieve_file}")
    return archieve_file

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

def get_latest_snippet_json_data(snippet_state: str, need_content=True, json_run_id_with_num=False, fetch_previous=False):
    """
    Retrieve the latest snippet content for a given state from the archive JSON.
    Returns (content, archive_dir, run_id).
    """
    info = get_last_run_info()
    run_id = info["run_id"]
    run_id_num = str(info["run_id_num"])  # ✅ ensure string key

    archive_dir = info.get("archive") or info.get("archieve")
    if not archive_dir:
        print("❌ Archive directory not found in get_last_run_info() result")
        return {}, archive_dir, run_id, run_id_num
        # return {"history": {}}, archive_dir, run_id, run_id_num

    archive_file = os.path.join(archive_dir, f"snippets_{run_id}.json")

    if not os.path.exists(archive_file):
        print(f"❌ Archive file {archive_file} not found.")
        return {}, archive_dir, run_id, run_id_num
        # return {"history": {}}, archive_dir, run_id, run_id_num

    try:
        with open(archive_file, "r", encoding="utf-8") as f:
            json_snippets = json.load(f)
    except json.JSONDecodeError:
        print(f"❌ Failed to parse JSON {archive_file}.")
        return {}, archive_dir, run_id, run_id_num

    history = json_snippets.get("history", {})
    if run_id not in history:
        print(f"❌ No entry for run_id {run_id} in {archive_file}.")
        # return {}, archive_dir, run_id, run_id_num
        return {}, archive_dir, run_id, run_id_num

    latest_entry = history[run_id]
    if run_id_num not in latest_entry:
        print(f"❌ No entry for run_id_num {run_id_num} in {archive_file}.")
        # return {}, archive_dir, run_id, run_id_num
        return {}, archive_dir, run_id, run_id_num

    section = latest_entry[run_id_num].get(snippet_state)
    if section:
        if need_content:
            # ✅ Prefer "content", fallback to "chunks"
            if "content" in section:
                return section["content"], archive_dir, run_id, run_id_num
            elif "chunks" in section:
                return section["chunks"], archive_dir, run_id, run_id_num
        elif json_run_id_with_num:
            runn_id_mit_num_prev_patch = {run_id: {run_id_num: section}}
            return runn_id_mit_num_prev_patch, archive_dir, run_id, run_id_num
        else:
            return section, archive_dir, run_id, run_id_num

    print(f"❌ Failed to get snippet '{snippet_state}' from run_id {run_id}.")
    return {}, archive_dir, run_id, run_id_num
    # return {"history": {}}, archive_dir, run_id, run_id_num

def store_history_snippets_json(snippet_name, snippet_data, chunks_only=True, need_token=True, model_name=None):
    # need consider store from latest_patch to previous_patch
    info = get_last_run_info()
    run_id = info["run_id"]
    run_id_num = str(info["run_id_num"])  # ensure string key
    archive_dir = info.get("archive") or info.get("archieve")

    if model_name is None:
        model_name = os.getenv("REASONING_AGENT")

    if not archive_dir or not os.path.exists(archive_dir):
        # archive_dir.mkdir(exist_ok=True)
        print(f"❌ Archive directory not found: {archive_dir}")
        return None

    archive_file = Path(archive_dir) / f"snippets_{run_id}.json"

    # Initialize JSON if file doesn't exist
    if archive_file.exists():
        try:
            data = json.loads(archive_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {"history": {}}
    else:
        data = {"history": {}}

    # ✅ Ensure "history" structure exists
    if "history" not in data:
        data["history"] = {}
    if run_id not in data["history"]:
        data["history"][run_id] = {}
    if run_id_num not in data["history"][run_id]:
        data["history"][run_id][run_id_num] = {}

    if snippet_data and snippet_name and chunks_only:
        snippet_entry = {
            "chunks": snippet_data if isinstance(snippet_data, list) else [snippet_data],
            "token": count_tokens(snippet_data, model_name, False) if need_token else None,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # if extras:
        #     if isinstance(extras, dict):
        #         snippet_entry["extras"] = extras
        #     else:
        #         raise ValueError(f"⚠️ extras is not a dict, got {type(extras)}. Ignoring extras.")

        # ✅ Store under section name
        data["history"][run_id][run_id_num][snippet_name] = snippet_entry

        print(f"✅ Added chunks to {snippet_name} at run_id {run_id}, run_id_num {run_id_num}")
    elif snippet_data and snippet_name and not chunks_only:
        data["history"][run_id][run_id_num][snippet_name] = snippet_data

        print(f"✅ Added JSON to {snippet_name} at run_id {run_id}, run_id_num {run_id_num}")
    else:
        raise ValueError(f"❌ {snippet_name} is None. Cannot store snippet.")

    # Save back
    archive_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Updated snippets archive at {archive_file}")
    return str(archive_file)

def load_json_safe(path):
    text = Path(path).read_text(encoding="utf-8")
    # remove illegal trailing commas before ] or }
    text = re.sub(r",(\s*[\]}])", r"\1", text)
    return json.loads(text)

def join_json_chunks(section: str = None, previous_enable = False) -> str:
    """
    Load chunks from a JSON archive for the last run_id.
    If section is provided, return only that section's chunks.
    If section is None, return all sections joined together.
    """
    info = get_last_run_info()
    run_id = info["run_id"]
    run_id_num = info["run_id_num"]

    # get the previous run_id and run_id_num
    if previous_enable:
        None

    archive_dir = info.get("archive") or info.get("archieve")
    if not archive_dir:
        raise KeyError("❌ 'archive' directory not found in get_last_run_info() result")

    json_file = os.path.join(archive_dir, f"snippets_{run_id}.json")

    if not os.path.exists(json_file):
        print(f"❌ Archive file {json_file} not found.")
        return ""

    # archive = json.loads(Path(json_file).read_text(encoding="utf-8"))
    archive = load_json_safe(Path(json_file))
    latest_data = archive.get("raw_data", {}).get(run_id, {})

    # ✅ Get the dict for this run_id_num
    # latest_data = run_data.get(run_id_num, {})
    if not latest_data:
        print(f"❌ No entry for run_id_num {run_id_num} in {json_file}.")
        return 

    if section:
        section_data = latest_data.get(section, {})
        chunks = section_data.get("chunks", [])
        return "\n".join(chunks)
    else:
        # Join all sections if none specified
        all_chunks = []
        for sec, sec_data in latest_data.items():
            all_chunks.extend(sec_data.get("chunks", []))
        return "\n".join(all_chunks)

def analyze_and_improve(
    ollama_server, 
    user_prompt, 
    snippets_enable=False, 
    json_chunks_enable=False, 
    query_text=None, 
    enable_llminvoke=True
):
    info = get_last_run_info()
    run_id = info.get("run_id")
    start = time.perf_counter()
    model_name = os.getenv("REASONING_AGENT")
    print("start analyze_and_improve")
    reasoning_prompt = ""

    if user_prompt:
        reasoning_prompt += f"""{user_prompt}"""

    if snippets_enable:
        data, archieve_dir, run_id = load_snippets(ollama_server, model_name)
        raw_data_json = data.get("raw_data", {})
        run_entries = raw_data_json.get(run_id, [])
        snippets_json = run_entries.get(run_id_num, {})

        # Flatten into one list of texts
        all_texts = []
        for section_name, section_data in snippets_json.items():
            section_tokens = section_data.get("total_tokens")
            if section_tokens is not None:
                print(f"Section {section_name} tokens: {section_tokens}")
            else:
                print(f"Section {section_name} has {len(section_data.get('content', []))} snippets")
            all_texts.extend(section_data.get("content", []))

        elapsed = time.perf_counter() - start
        total_tokens = count_tokens(all_texts, model_name, False)
        print(f"retrieved {len(all_texts)} snippets with Total tokens of {total_tokens} in {elapsed:.2f} seconds")

        # Inspect snippets AFTER retrieval
        inspect_snippets(
            [{"section": sec, "chunks": txt}
             for sec, sec_data in snippets_json.items()
             for txt in sec_data.get("chunks", [])],
            expected_sections=["table_deals", "mq5_code", "mqh_header", "log"]
        )

        def join_content(key):
            return "\n".join(snippets_json.get(key, {}).get("chunks", []))

        reasoning_prompt += f"""
        Context:
        table_deals:
        {join_content("table_deals")}
        table_orders:
        {join_content("table_orders")}
        table_results:
        {join_content("table_results")}
        mq5_code:
        {join_content("mq5_code")}
        mqh_header:
        {join_content("mqh_header")}
        log:
        {join_content("log")}
        """

    elif json_chunks_enable:
        reasoning_prompt += f"""
        Context:
        table_deals:
        {join_json_chunks("table_deals")}
        table_orders:
        {join_json_chunks("table_orders")}
        table_results:
        {join_json_chunks("table_results")}
        mq5_code:
        {join_json_chunks("mq5_code")}
        mqh_header:
        {join_json_chunks("mqh_header")}
        backtested_log:
        {join_json_chunks("log")}
        """

        # check if any latest_patch and 
        # if 

    analysis = ""
    # Always run analysis
    if enable_llminvoke:
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
            explanation_code = parsed.get("explanation_code", "")
            analysis_code = parsed.get("analysis_code", "")
            # Remove JSON from analysis text
            analysis_text = response.replace(json_str, "").strip()
    except Exception as e:
        print("JSON parse error:", e)

    return {
        "analysis_code": analysis_code,
        "old_code": old_code,
        "fix_code": fix_code,
        "explanation_code": explanation_code
    }

def suggest_code_improvements(
    ollama_server,
    last_reasoning_prompt,
    analysis,
    coder_prompt,
    fetch_snippets_enable=False,
    compile_error_enable=False,
    compile_error=None,
    patch_files_list=None
):
    print("start executing suggest_code_improvements ...")

    if patch_files_list is None:
        patch_files_list = []

    # --- Guard clause: check required inputs ---
    missing = []
    if not last_reasoning_prompt:
        missing.append("last_reasoning_prompt")
    if not analysis:
        missing.append("analysis")
    if not coder_prompt:
        missing.append("coder_prompt")

    info = get_last_run_info()
    run_id = info["run_id"]
    run_id_num = info["run_id_num"]

    if missing:
        if fetch_snippets_enable:
            print("⚠️ Missing required inputs:", ", ".join(missing), ". Loading from FAISS snippets...")
            model_name = os.getenv("CODER_AGENT")
            data, archive_dir, run_id = load_snippets(ollama_server, model_name)

            raw_data_json = data.get("raw_data", {})
            run_entries = raw_data_json.get(run_id, {})
            snippets_json = run_entries.get(run_id_num, {})

            if not last_reasoning_prompt and "reasoning_prompt" in snippets_json:
                last_reasoning_prompt = "\n".join(snippets_json["reasoning_prompt"].get("chunks", []))
            if not analysis and "analysis" in snippets_json:
                analysis = "\n".join(snippets_json["analysis"].get("chunks", []))
        else:
            if "analysis" in missing:
                analysis, _, _, _ = get_latest_snippet_json_data("analysis")

    # --- Error if still missing ---
    if not last_reasoning_prompt:
        raise ValueError("❌ reasoning_prompt is None. Cannot proceed.")
    if not analysis:
        raise ValueError("❌ analysis is None. Cannot proceed.")
    if not coder_prompt:
        raise ValueError("❌ coder_prompt is None. Cannot proceed.")

    # --- Snippet retrieval ---
    if fetch_snippets_enable:
        print("fetch full snippets for coder and header ...")
        start = time.perf_counter()
        code_snippets   = query_last_run_snippets(
            ollama_server, doc_type="code", section="mq5_code", top_k=100
        )[0]
        header_snippets = query_last_run_snippets(
            ollama_server, doc_type="header", section="mqh_header", top_k=100
        )[0]
        snippets = code_snippets + header_snippets
        elapsed = time.perf_counter() - start
        print(f"retrieved {len(snippets)} snippets in {elapsed:.2f} seconds")
    else:
        code_snippets = join_json_chunks("mq5_code").splitlines()
        header_snippets = join_json_chunks("mqh_header").splitlines()
        snippets = code_snippets + header_snippets

    # --- Build coder prompt ---
    reasoning_coder_prompt = f"""Based on last reasoning prompt:\n{last_reasoning_prompt}\n
    and based on last reasoning prompt analysis result:\n{analysis}\n
    """

    if fetch_snippets_enable:
        reasoning_coder_prompt += f"""
    mql5 source code: {" ".join(str(s) for s in code_snippets)}\n
    header source code: {" ".join(str(s) for s in header_snippets)}\n
    """
    else:
        reasoning_coder_prompt += f"""
    mq5_code:\n{join_json_chunks("mq5_code")}\n
    mqh_header:\n{join_json_chunks("mqh_header")}\n
    """

    if compile_error_enable and compile_error:
        reasoning_coder_prompt += f"""
    patch files list: {patch_files_list}\n
    metatrader 5 compile errors: {compile_error}\n
    """

    reasoning_coder_prompt += f"""
    {coder_prompt}\n
    """

    print("generating code ...")
    start = time.perf_counter()
    model_name = os.getenv("CODER_AGENT")
    code_response = safe_invoke_ollama(reasoning_coder_prompt, model_name, ollama_server)
    elapsed = time.perf_counter() - start
    print(f"code_response: {code_response} with {elapsed:.2f} seconds")

    return parse_response_with_analysis(code_response), reasoning_coder_prompt

def patch_funct_file(file_path, func_names, new_code):
    if not os.path.exists(file_path):
        return False
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="utf-16") as f:
            content = f.read()

    updated_content = content
    changed = False

    for func_name in func_names:
        target_pattern = rf"(?:void|int|double|string|bool|float)\s+{func_name}\s*\([^)]*\)\s*\{{.*?\}}"
        if re.search(target_pattern, updated_content, flags=re.S):
            new_func_def = re.search(target_pattern, new_code, flags=re.S)
            if new_func_def:
                updated_content = re.sub(target_pattern, new_func_def.group(0),
                                         updated_content, flags=re.S)
                print(f"✅ Updated {func_name}() in {file_path}")
                changed = True
        else:
            print(f"⚠️ Function {func_name} not found in {file_path}")

    if changed:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(updated_content)
        return True
    return False

def find_patch_history(base_path=""):
    """
    Iterate through memory.json RUN_HISTORY from top to bottom.
    For each run_id, check logs/{run_id}/snippets_{run_id}.json.
    If snippet contains latest_patch or previous_patch, return result.
    """
    memory_file = Path(base_path) / "memory.json"
    results = []

    try:
        # Load memory.json
        if not memory_file.is_file():
            raise FileNotFoundError(f"memory.json not found at {memory_file}")

        with memory_file.open("r", encoding="utf-8") as f:
            memory_data = json.load(f)

        run_history_list = memory_data.get("RUN_HISTORY", [])

        # Iterate top to bottom
        for run_entry in run_history_list:
            run_id = run_entry.get("run_id")
            if not run_id:
                continue

            snippet_file = Path(base_path) / "logs" / run_id / f"snippets_{run_id}.json"
            if not snippet_file.is_file():
                continue

            with snippet_file.open("r", encoding="utf-8") as f:
                snippet_data = json.load(f)

            # ✅ Handle new structure: raw_data[run_id] is a list of entries
            run_entries = snippet_data.get("raw_data", {}).get(run_id, [])
            if isinstance(run_entries, list) and run_entries:
                latest_entry = run_entries[0]   # newest entry if you insert(0,...)
                snippets_json = latest_entry.get("snippets", {})
            else:
                snippets_json = {}

            latest_patch = snippets_json.get("latest_patch", {}).get("content")
            previous_patch = snippets_json.get("previous_patch", {}).get("content")

            if latest_patch or previous_patch:
                results.append({
                    "run_id": run_id,
                    "latest_patch": latest_patch or "",
                    "previous_patch": previous_patch or ""
                })

    except Exception as e:
        return {"error": str(e)}

    return results

def generate_patch_from_git(code_repo, new_code, ea_file_path, header_file_path, mt5_mql5_dir, user_commit_message=None):
    """
    Apply function updates from new_code into EA (.mq5) or header (.mqh) files.
    Detects function names automatically and patches them in the correct file.
    """
    if not new_code or not new_code.strip():
        raise ValueError("❌ new_code is empty. Nothing to patch.")

    info = get_last_run_info()
    run_id = info["run_id"]
    run_id_num = info["run_id_num"]
    if not user_commit_message:
        user_commit_message = f"{run_id}_{run_id_num}"

    code_repo_ea_file = Path(code_repo) / ea_file_path
    code_repo_header_file = Path(code_repo) / header_file_path

    # --- Ensure repo is initialized ---
    if not (Path(code_repo) / ".git").exists():
        print(f"No Git repo found in {code_repo}. Initializing new repo...")
        Path(code_repo).mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init"], cwd=code_repo, check=True)

        # Copy baseline files
        for src, dest in [
            (Path(mt5_mql5_dir) / ea_file_path, code_repo_ea_file),
            (Path(mt5_mql5_dir) / header_file_path, code_repo_header_file),
        ]:
            if not dest.is_file() and src.is_file():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dest)
                print(f"Copied {src} → {dest}")

        # Add .gitignore
        gitignore_path = Path(code_repo) / ".gitignore"
        if not gitignore_path.exists():
            gitignore_path.write_text("*.patch\npatches\n", encoding="utf-8")

        subprocess.run(["git", "add", "."], cwd=code_repo, check=True)
        subprocess.run(["git", "commit", "-m", "init with original files"], cwd=code_repo, check=True)

    # --- Extract functions ---
    func_pattern = r"(?:void|int|double|string|bool|float)\s+[A-Za-z_]\w*\s*\([^)]*\)\s*\{.*?\}"
    new_funcs = re.findall(func_pattern, new_code, flags=re.S)
    name_pattern = r"\b(?:void|int|double|string|bool|float)\s+([A-Za-z_]\w*)\s*\("
    func_names = re.findall(name_pattern, new_code)

    if not func_names:
        print("❌ No function definitions detected in new_code.")
        return False

    print("Functions detected in new_code:", ", ".join(func_names))

    changed_files = []
    if patch_funct_file(code_repo_ea_file, func_names, new_code):
        changed_files.append(code_repo_ea_file)
    if patch_funct_file(code_repo_header_file, func_names, new_code):
        changed_files.append(code_repo_header_file)

    if not changed_files:
        raise RuntimeError("❌ No functions patched. Check file paths or function names.")

    # --- Commit changes ---
    # for file in changed_files:
        # subprocess.run(["git", "add", str(file)], cwd=code_repo, check=True)

        # --- Commit changes ---
        # --- Stage changes ---
    for file in changed_files:
        rel_path = os.path.relpath(file, code_repo)  # ensure relative path
        if not Path(file).is_file():
            raise RuntimeError(f"❌ File {file} not found or not modified, cannot add to git.")
        subprocess.run(["git", "add", rel_path], cwd=code_repo, check=True)

    # --- Only commit if there are staged changes ---
    status_result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=code_repo,
        capture_output=True,
        text=True
    )

    if status_result.stdout.strip():  # non-empty means there are changes
        result = subprocess.run(
            ["git", "commit", "-m", user_commit_message],
            cwd=code_repo,
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Git commit failed: {result.stderr}")
    else:
        print("⚠️ No changes detected, skipping commit.")
        return None

    if result.returncode != 0:
        raise RuntimeError(f"Git commit failed: {result.stderr}")

    # --- Generate patch ---
    patch_dir = Path(code_repo) / "patches"
    patch_dir.mkdir(exist_ok=True)
    patch_path = patch_dir / f"{user_commit_message}.patch"

    with patch_path.open("w", encoding="utf-8") as patch_file:
        subprocess.run(["git", "format-patch", "-1", "HEAD", "--stdout"],
                       cwd=code_repo, check=True, stdout=patch_file, text=True)

    # Reset repo back to baseline (optional)
    subprocess.run(["git", "reset", "--hard", "HEAD~1"], cwd=code_repo, check=True)

    return str(patch_path)

def get_patch_content(patch_path: str) -> str:
    """
    Read and return the content of a patch file.
    Returns the file content as a string, or an empty string if not found.
    """
    if not patch_path:
        return ""

    if os.path.isfile(patch_path):
        try:
            with open(patch_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            # Return error message if file can't be read
            return f"Error reading patch file: {e}"
    else:
        return ""

def commit_exists(repo_dir, commit_sha):
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{commit_sha}^{{commit}}"],
            cwd=repo_dir,
            check=True,
            capture_output=True
        )
        return True
    except subprocess.CalledProcessError:
        return False

def get_patch_commit_sha(patch_file):
    with open(patch_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("From "):
                return line.split()[1]
    return None

def verify_latest_patch_commit(repo_dir, patch_file):
    """
    Verify that the latest commit message matches the patch file name.
    Returns True if they match, False otherwise.
    """
    # Get patch filename (without extension)
    patch_name = os.path.splitext(os.path.basename(patch_file))[0]

    # Get latest commit message
    result = subprocess.run(
        ["git", "log", "-1", "--pretty=%B"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=True
    )
    latest_commit_message = result.stdout.strip()

    print(f"Patch name: {patch_name}")
    print(f"Latest commit message: {latest_commit_message}")

    return latest_commit_message == patch_name

def apply_patch_to_git(repo_dir, patch_file, github_push=False):
    if not os.path.exists(os.path.join(repo_dir, ".git")):
        raise RuntimeError(f"❌ No git repo in {repo_dir}.")

    patch_file = os.path.abspath(patch_file)
    commit_sha = get_patch_commit_sha(patch_file)

    if commit_sha and commit_exists(repo_dir, commit_sha) and verify_latest_patch_commit(repo_dir, patch_file):
        print(f"⚠️ Commit {commit_sha} already exists in repo, skipping git am.")
    else:
        try:
            patch_rel_path = os.path.relpath(patch_file, repo_dir)
            subprocess.run(["git", "am", patch_rel_path], cwd=repo_dir, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"❌ Failed to apply patch {patch_rel_path}: {e}")

    # --- Get commit message from last commit ---
    result = subprocess.run(
        ["git", "log", "-1", "--pretty=%B"],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=True
    )
    full_commit_message = result.stdout.strip()

    print(f"✅ Improvements applied and committed: {full_commit_message}")

    # --- Push to GitHub if requested ---
    if github_push:
        github_url = os.getenv("GITLAB_EA_REMOTE")  # or GITHUB_EA_REMOTE if you prefer
        try:
            # Ensure remote is set
            subprocess.run(["git", "remote", "set-url", "origin", github_url], cwd=repo_dir, check=True)
            subprocess.run(["git", "push", "origin", "HEAD"], cwd=repo_dir, check=True)
            print("🌐 Changes pushed to GitHub.")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"❌ Git push failed: {e}")

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
                    cleaned = " ".join(parts[5:])
                    outfile.write(cleaned + "\n")
            print(f"Cleaned log written to {output_file} (read as {enc})")
            return output_file
        except UnicodeDecodeError:
            continue

    raise ValueError("Failed to decode file with common encodings.")


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




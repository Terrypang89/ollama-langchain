import streamlit as st
import subprocess
import difflib
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from testMT5 import update_ini_file, run_mt5_backtest, compile_ea, safe_read_file, \
report_tables_to_json, process_run_for_embeddings, load_params_from_ini, \
save_run_and_update_memory, generate_patch_from_git, query_last_run_snippets, \
extract_tester_report_summary, analyze_and_improve, clean_log

load_dotenv()
OLLAMA_SERVER = os.getenv("OLLAMA_API_BASE")
ollama_model = OllamaLLM(model="mistral:7b", base_url=OLLAMA_SERVER)

def preview_improvements(old_code_file, code_repo, improvements, commit_message):
    st.title("Backtest EA Review")

    st.subheader("Compile Expert Advisor")
    if st.button("Compile EA"):
        try:
            AGENT_PATH = os.getenv("AGENT_PATH")
            EA_MQ5_SUBPATH = os.getenv("EA_MQ5_SUBPATH")
            EA_MQL_FILE = os.path.join(AGENT_PATH, EA_MQ5_SUBPATH)
            MT5_PATH = os.getenv("MT5_PATH")
            METAEDITOR = os.path.join(MT5_PATH, os.getenv("METAEDITOR_SUBPATH"))
            result = compile_ea(EA_MQL_FILE, METAEDITOR)
            st.success(f"EA compiled successfully: {result}")
        except Exception as e:
            st.error(f"Compilation failed: {e}")

    # --- Run MT5 Backtest button ---
    st.subheader("Run MT5 Backtest")
    if st.button("Run Backtest"):
        try:
            # Auto-generate ini file before running backtest
            EA_EX_NAME = os.getenv("EA_EX_NAME")
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
                report_path="Tester_report.html",
                updates={
                    "Expert": EA_EX_NAME,
                    "Symbol": "XAUUSD",
                    "Period": "M5",
                    "FromDate": "2025.03.01",
                    "ToDate": "2025.03.20",
                    "Visual": True,
                    "Report": "Tester_report.html",
                }
            )

            # Run backtest immediately after ini generation
            MT5_PATH = os.getenv("MT5_PATH")
            TERMINAL_PATH = os.path.join(MT5_PATH, os.getenv("TERMINAL_SUBPATH"))
            AGENT_PATH = os.getenv("AGENT_PATH")
            BACKTEST_REPORT_PATH = AGENT_PATH
            BACKTEST_LOG_PATH = os.path.join(AGENT_PATH, os.getenv("BACKTEST_LOG_SUBPATH"))
            # add column to insert portable, timeout and visual
            run_id, archive_folder, report_file, log_file = run_mt5_backtest(
                config_path=ini_file,
                terminal_path=TERMINAL_PATH,
                report_path=BACKTEST_REPORT_PATH,
                log_path=BACKTEST_LOG_PATH,
                store_path="logs",
                portable_enable=False,
                timeout=60,
            )

            # Convert report tables to JSON
            json_report_file = report_tables_to_json(
                report_file,
                archive_folder=archive_folder,
                output_json="report_tables.json",
            )

            # clean log file
            clean_log_file = clean_log(log_file, archive_folder=archive_folder)

            # Parse summary from JSON
            # summary = parse_backtest_report(json_report_file)
            summary,json_report_file = extract_tester_report_summary(json_report_file)

            # Display summary nicely
            st.subheader("Backtest Summary")
            st.json(summary)

            st.success(f"Backtest completed! Run ID: {run_id}")
            st.write(f"Archive folder: {archive_folder}")
            st.write(f"Report file: {json_report_file}")
            st.write(f"Log file: {clean_log_file}")

             # Use safe_read_file to avoid encoding errors
            # report_file = st.session_state.get("report_file")
            if os.path.exists(report_file):
                report_content = safe_read_file(report_file)
                st.download_button("Download Report", report_content, file_name="Tester_report.html")
            # log_file = st.session_state.get("log_file")
            if os.path.exists(clean_log_file):
                log_content = safe_read_file(clean_log_file)
                st.download_button("Download Log", log_content, file_name="Tester_log.txt")

            # Store results in session state for later use
            st.session_state["run_id"] = run_id
            st.session_state["archive_folder"] = archive_folder
            st.session_state["json_report_file"] = json_report_file
            st.session_state["clean_log_file"] = clean_log_file
            st.session_state["ini_file"] = ini_file
            st.session_state["summary"] = summary

        except Exception as e:
            st.error(f"Backtest failed: {e}")
        
    # --- Separate button for storing to vector store ---
    st.subheader("Persist Run to Vector Store")
    if st.button("Store to Vector Store"):
        try:
            AGENT_PATH = os.getenv("AGENT_PATH")
            EA_MQ5_SUBPATH = os.getenv("EA_MQ5_SUBPATH")
            EA_HEADER_SUBPATH = os.getenv("EA_HEADER_SUBPATH")
            EA_MQL_FILE = os.path.join(AGENT_PATH, EA_MQ5_SUBPATH)
            HEADER_MQL_FILE = os.path.join(AGENT_PATH, EA_HEADER_SUBPATH)
            EA_EX_NAME = os.getenv("EA_EX_NAME")

            run_id = st.session_state.get("run_id")
            archive_folder = st.session_state.get("archive_folder")
            json_report_file = st.session_state.get("json_report_file")
            clean_log_file = st.session_state.get("clean_log_file")
            ini_file = st.session_state.get("ini_file")

            if not all([run_id, archive_folder, json_report_file, clean_log_file, ini_file]):
                st.warning("Please run a backtest first before storing to vector store.")
            else:
                st.write(f"Performing process_run_for_embeddings with run_id: {run_id} ...")
                vector_ids = process_run_for_embeddings(
                    ollama_server=OLLAMA_SERVER,
                    run_id=run_id,
                    ea_source_file=EA_MQL_FILE,
                    log_file=clean_log_file,
                    report_file=json_report_file,
                    header_files=HEADER_MQL_FILE,
                )
                st.success(f"Run stored to vector store! Vector IDs: {vector_ids}")
                st.write(f"Performing save_run_and_update_memory ...")

                metadata_path = save_run_and_update_memory(
                    run_id=run_id,
                    parameters=load_params_from_ini(ini_file),
                    archive_folder=archive_folder,
                    report_files=[os.path.basename(json_report_file)],
                    log_file=os.path.basename(clean_log_file),
                    vector_ids=vector_ids
                )

                st.success(f"save_run_and_update_memory done! Metadata saved at: {metadata_path}")
                st.write(f"Vector IDs: {vector_ids}")
        except Exception as e:
            st.error(f"Failed to store run to vector store: {e}")

    # --- Load old code from file ---
    if not os.path.exists(old_code_file):
        st.error(f"File not found: {old_code_file}")
        return
    with open(old_code_file, "r", encoding="utf-8") as f:
        old_code = f.read()

    # --- Improvements summary ---
    st.subheader("Analyze Improve Summary")
    if st.button("Analyze && Improve "):
        fix_data = analyze_and_improve(OLLAMA_SERVER)
        st.write(f"reasoning_prompt: {fix_data["reasoning_prompt"]}")
        st.write(f"analysis: {fix_data["analysis"]}")
        st.write(f"code_fix: {fix_data["code_fix"]}")

    # --- Suggestions box ---
    st.subheader("Suggest Code Changes")
    user_suggestions = st.text_area("Your Suggestions", "", height=200)

    # --- Regenerate button ---
    if st.button("Regenerate Code Changes"):
        if user_suggestions.strip() or improvements.strip():
#             prompt = f"""Here is the original code:\n{old_code}\n
# Apply these improvements:\n{improvements}\n
# And also consider these suggestions:\n{user_suggestions}\n
# Please output the full updated code without commented:"""

            # response = ollama_model.invoke(prompt)
            # response = suggest_code_improvements(OLLAMA_SERVER, user_suggestions)
            # code_suggest = "simplify my code"
            snippets = query_last_run_snippets(OLLAMA_SERVER, user_suggestions)

            print("snippets:", snippets)

            # Step 2: Build context for the model
            context = "\n\n".join([
                f"[{s.metadata.get('doc_type','unknown')} snippet]\n{s.page_content}"
                for s in snippets
            ])

            # Step 3: Send to model for improvement suggestions
            prompt = f"""
        You are an expert MQL5 developer. Analyze the following snippets (code, headers, logs, reports)
        and suggest improvements to the EA source code. Specially trading strategy. 

        Query: {user_suggestions}

        Context:
        {context}

        Provide specific MQ5 code changes or refactoring ideas. Then output the full updated code.
            """

            ollama_model = OllamaLLM(model="mistral:7b", base_url=OLLAMA_SERVER)
            response = ollama_model.invoke(prompt)

            st.session_state["new_code"] = response
            st.success("Code changes regenerated using Ollama!")
        else:
            st.warning("Please enter improvements or suggestions before regenerating.")

    # --- Side-by-side preview ---
    st.subheader("Side-by-Side Code Preview")
    col1, col2 = st.columns(2)
    with col1:
        st.code(old_code, language="python")
    with col2:
        st.code(st.session_state.get("new_code", old_code), language="python")

    if "new_code" in st.session_state:
        if st.button("Download Patch File"):
            try:
                patch_path = generate_patch_from_git(
                    code_repo,
                    old_code_file,
                    user_commit_message,
                    st.session_state["new_code"]
                )
                st.success(f"Patch file generated: {patch_path}")
                st.session_state["latest_patch_path"] = patch_path
                st.info("You can now apply it using the Apply button below.")
            except RuntimeError as e:
                st.error(str(e))

    if st.button("Apply Approved Hunks to Git"):
        patch_path = st.session_state.get("latest_patch_path")
        if not patch_path or not os.path.exists(patch_path):
            patch_dir = os.path.join(code_repo, "patches")
            patch_files = [f for f in os.listdir(patch_dir) if f.endswith(".patch")]
            if not patch_files:
                st.error("No patch file found. Please generate one first.")
                patch_path = None
            else:
                patch_files.sort()
                latest_patch = patch_files[-1]
                patch_path = os.path.join(patch_dir, latest_patch)
                st.info(f"Using latest patch file: {patch_path}")

        if patch_path and os.path.exists(patch_path):
            try:
                patch_relpath = os.path.relpath(patch_path, code_repo).replace("\\", "/")
                subprocess.run(["git", "am", patch_relpath], cwd=code_repo, check=True)
                st.success(f"Patch applied successfully from {patch_relpath}!")
            except subprocess.CalledProcessError as e:
                st.error(f"Git error while applying patch: {e}")


if __name__ == "__main__":
    # Example old vs new code

    improvements = "Added type safety to add() function."
    commit_message = "Enhance add() with type safety"

    preview_improvements("test_code\Testcode.py", "test_code", improvements, commit_message)


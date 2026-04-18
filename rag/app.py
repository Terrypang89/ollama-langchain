import streamlit as st
import subprocess
import difflib
import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from cleanstore import clean_store, clean_snippets_json
from testMT5 import update_ini_file, run_mt5_backtest, copyfiles, compile_ea, extract_errors, count_tokens, \
report_tables_to_json, load_params_from_ini, store_history_snippets_json, apply_patch_to_git, get_patch_content, process_run_for_embeddings, \
save_run_and_update_memory, generate_patch_from_git, get_latest_snippet_json_data, beautify_text_area, find_patch_history, \
extract_tester_report_summary, analyze_and_improve, clean_log, suggest_code_improvements, compile_fail_update_memory

load_dotenv()
OLLAMA_SERVER = os.getenv("OLLAMA_API_BASE")

def preview_improvements():
    st.title("Backtest EA Review")

    st.subheader("Copy source files to MT5")
    with st.form("copyfiles_form"):
        # You can add inputs here if needed, e.g. text boxes for paths
        copyfiles_submitted = st.form_submit_button("Copy files")
        if copyfiles_submitted:
            try:
                commit_copyfiles = copyfiles(
                    os.getenv("EA_CODE_GIT_REPO"), 
                    os.getenv("EA_MQ5_SUBPATH"), 
                    os.getenv("EA_HEADER_SUBPATH"), 
                    os.getenv("AGENT_PATH")
                )
                st.session_state["commit_copyfiles"] = commit_copyfiles
                # Store history if commit info is available
                store_history_snippets_json(
                    "copyfiles",
                    f"Files copied from {os.getenv('EA_CODE_GIT_REPO')} "
                    f"to {os.getenv('AGENT_PATH')} successfully "
                    f"with latest commit: {st.session_state["commit_copyfiles"]}"
                )
            except Exception as e:
                st.error(f"❌ Copy operation failed: {e}")

        if st.session_state.get("commit_copyfiles"):
            st.success(
                f"✅ Files copied from {os.getenv('EA_CODE_GIT_REPO')} "
                f"to {os.getenv('AGENT_PATH')} successfully "
                f"with latest commit: {st.session_state["commit_copyfiles"]}"
            )
        # else:
        #     st.warning("⚠️ Files copy failed or no commit info available.")

    st.subheader("Compile Expert Advisor")
    with st.form("compile_ea_form"):
        submitted = st.form_submit_button("Compile EA")
        if submitted:
            try:
                compile_result = compile_ea(
                    os.path.join(os.getenv("AGENT_PATH"), os.getenv("EA_MQ5_SUBPATH")), 
                    os.path.join(os.getenv("MT5_PATH"), os.getenv("METAEDITOR_SUBPATH"))
                )
                # Extract only error lines
                compile_error = extract_errors(compile_result)
                st.session_state['compile_result'] = compile_result
                store_history_snippets_json("compile_result", st.session_state['compile_result'])

                if compile_error:
                    st.session_state['compile_error'] = compile_error
                    store_history_snippets_json("compile_error", st.session_state['compile_error'])
                    st.error("❌ Compilation errors found:")
                    # Increment LAST_RUN_ID_NUM on failure
                    new_run_id_num = compile_fail_update_memory()
                    st.session_state['run_id_num'] = new_run_id_num
                    st.info(f"Memory updated: LAST_RUN_ID_NUM = {new_id_num}")
                    for err in compile_error:
                        st.write(err)
                else:
                    st.success("✅ EA compiled successfully (no errors)")
                    if "warning" in compile_result.lower():
                        st.warning("⚠️ Compilation completed with warnings")
                        # st.text(compile_result)

            except Exception as e:
                st.error(f"❌ Compilation failed: {e}")

        # Show results after form submission
        if st.session_state.get("compile_result"):
            st.write(f"Overall result: \n {st.session_state['compile_result']}")

    # --- Run MT5 Backtest button ---
    st.subheader("Run MT5 Backtest")
    with st.form("edit_backtest_form"):
        col1, col2, _ = st.columns([1.5 ,1, 10])  # adjust ratio for width
        with col1:
            visual_enable = st.checkbox("🔍 Include Visual")
        with col2:
            timeout_value = st.number_input("⏱ Timeout (s)", min_value=5, max_value=5000, value=60, step=10)
        run_backtest = st.form_submit_button("💾 Run Backtest")
        if run_backtest:
            try:
                # perform copy previous_patch and latest_patch to session state
                if not st.session_state.get("latest_patch"):
                    st.session_state["latest_patch"], _, _, _ = get_latest_snippet_json_data("latest_patch", False)
                if not st.session_state.get("previous_patch_json"):
                    st.session_state["previous_patch_json"], _, _, _ = get_latest_snippet_json_data("previous_patch", False)

                # Auto-generate ini file before running backtest
                ini_file = update_ini_file(
                    ini_path="tester.ini",
                    login=os.getenv("MT5_LOGIN"),
                    password=os.getenv("MT5_PASSWORD"),
                    server=os.getenv("MT5_SERVER"),
                    expert=os.getenv("EA_EX_NAME"),
                    symbol="XAUUSD",
                    period="M5",
                    from_date="2025.03.01",
                    to_date="2025.03.20",
                    deposit=10000,
                    currency="USD",
                    leverage="1:100",
                    visual=visual_enable,
                    report_path="Tester_report.html",
                )

                # add column to insert portable, timeout and visual
                run_id, archieve_folder, report_file, log_file = run_mt5_backtest(
                    config_path=ini_file,
                    terminal_path=os.path.join(os.getenv("MT5_PATH"), os.getenv("TERMINAL_SUBPATH")),
                    report_path=os.getenv("AGENT_PATH"),
                    log_path=os.path.join(os.getenv("AGENT_PATH"), os.getenv("BACKTEST_LOG_SUBPATH")),
                    store_path="logs",
                    portable_enable=False,
                    timeout=timeout_value,
                )

                # Convert report tables to JSON
                json_report_file = report_tables_to_json(
                    report_file,
                    archieve_folder=archieve_folder,
                    output_json="report_tables.json",
                )

                # clean log file
                clean_log_file = clean_log(log_file, archieve_folder=archieve_folder)

                # Parse summary from JSON
                summary, json_report_file = extract_tester_report_summary(json_report_file)

                # Store results in session state for later use
                st.session_state["run_id"] = run_id
                st.session_state["archieve_folder"] = archieve_folder
                st.session_state["json_report_file"] = json_report_file
                st.session_state["clean_log_file"] = clean_log_file
                st.session_state["ini_file"] = ini_file
                st.session_state["summary"] = summary

            except Exception as e:
                st.error(f"Backtest failed: {e}")

        if st.session_state.get("summary"):
            # Display summary nicely
            st.subheader("Backtest Summary")
            st.json(st.session_state.get("summary"))
            st.success(f"Backtest completed! New Run ID: {st.session_state.get("run_id")}")
            st.write(f"Archive folder: {st.session_state.get("archieve_folder")}")
            st.write(f"Report file: {st.session_state.get("json_report_file")}")
            st.write(f"Log file: {st.session_state.get("clean_log_file")}")
            print("Edits are saved in session state.")

    st.subheader("Persist Run to Vector Store")
    with st.form("vector_store_form"):
        faiss_embedded_enable = st.checkbox("🔍 FAISS mebedded", value=True)
        Store_submitted = st.form_submit_button("Store to Vector Store")

        if Store_submitted:
            try:
                run_id = st.session_state.get("run_id")
                archieve_folder = st.session_state.get("archieve_folder")
                json_report_file = st.session_state.get("json_report_file")
                clean_log_file = st.session_state.get("clean_log_file")
                ini_file = st.session_state.get("ini_file")

                if not all([run_id, archieve_folder, json_report_file, clean_log_file, ini_file]):
                    st.warning("⚠️ Please run a backtest first before storing to vector store.")
                else:
                    st.write(f"Performing process_run_for_embeddings with run_id: {run_id} ...")
                    vector_ids = process_run_for_embeddings(
                        ollama_server=OLLAMA_SERVER,
                        run_id=run_id,
                        archieve_folder=archieve_folder,
                        ea_source_file=os.path.join(os.getenv("AGENT_PATH"), os.getenv("EA_MQ5_SUBPATH")),
                        log_file=clean_log_file,
                        report_file=json_report_file,
                        header_files=os.path.join(os.getenv("AGENT_PATH"), os.getenv("EA_HEADER_SUBPATH")),
                        faiss_embedded=faiss_embedded_enable
                    )
                    st.success(f"✅ Run stored to vector store! Vector IDs: {vector_ids}")
                    st.session_state["vector_store"] = vector_ids

                    st.write("Performing save_run_and_update_memory ...")
                    metadata_path = save_run_and_update_memory(
                        run_id=run_id,
                        parameters=load_params_from_ini(ini_file),
                        archieve_folder=archieve_folder,
                        report_files=[os.path.basename(json_report_file)],
                        log_file=os.path.basename(clean_log_file),
                        vector_ids=vector_ids
                    )

                    # perform the copy 
                    if st.session_state.get("latest_patch"):
                        store_history_snippets_json("latest_patch", st.session_state["latest_patch"], False)
                    if st.session_state.get("previous_patch_json"):
                        store_history_snippets_json("previous_patch",st.session_state["previous_patch_json"], False)

                    # start new session
                    st.session_state.clear()
                    st.success("All session_state cleared!")
                    print("All session_state cleared!")

                    st.success(f"✅ Metadata saved at: {metadata_path}")
                    st.session_state["Metadata"] = metadata_path
                    st.write(f"Vector IDs: {vector_ids}")
            except Exception as e:
                st.error(f"❌ Failed to store run to vector store: {e}")

        if st.session_state.get("vector_store") and st.session_state.get("Metadata"):
            st.info(f" stored to vector store! Vector IDs: {st.session_state["vector_store"]}, Metadata saved at: {st.session_state["Metadata"]}")

    # --- Improvements summary ---
    st.subheader("Analyze Improve Summary")
    with st.form("edit_analyze_data_form"):
        prompt_text = f"""You are an expert MQL5 developer and quantitative trader.
            Analyze the loss deals, resolve with EA improvements, retest, and check differences.

            Tasks:
            1) Study the mq5_code and mqh_header, then match to log wheres log is from metatrader 5 backtested result. 
            2) Identify all loss deals in report_tables_clean table_deals from context.
            3) For each loss deal, match the deal time with context log entries and extract the reason.
            4) Inspect the EA function Trade_Strategy. Advise how to resolve the issue.
            5) Propose specific MQL5 code changes or refactors in Trade_Strategy (show code snippets).
            """
        prompt_suggestions = st.text_area("Prompt", prompt_text, height=300)
    
        # Checkbox to decide whether to fetch snippets
        col1, col2, _ = st.columns([2, 2, 10], gap="xxsmall")
        with col1:
            snippets_enable = st.checkbox("🔍 Include FAISS snippets", value=False)

        with col2:
            json_chunk_enable = st.checkbox("🔍 Include JSON chunks", value=True)
        
        analyze_submitted = st.form_submit_button("Analyze && Improve")

        if analyze_submitted:
            fix_data = analyze_and_improve(OLLAMA_SERVER, prompt_suggestions, snippets_enable, json_chunk_enable)
            st.session_state["prompt_suggestions"] = prompt_suggestions
            st.session_state["reasoning_prompt"] = fix_data["reasoning_prompt"]
            st.session_state["analysis"] = fix_data["analysis"]
            st.session_state["model_name"] = fix_data["model_name"]
            st.session_state["run_id"] = fix_data["run_id"]

        # if see nothing, then obtain from json file
        if not st.session_state.get("reasoning_prompt"):
            st.session_state["reasoning_prompt"], _, _, _ = get_latest_snippet_json_data("reasoning_prompt")

        reasoning_prompt_edit = st.text_area(
            "Reasoning Prompt (editable)", \
            value=beautify_text_area(st.session_state.get("reasoning_prompt")), \
            height=300
        )

        # if see nothing, then obtain from json file
        if not st.session_state.get("analysis"):
            st.session_state["analysis"], _, _, _ = get_latest_snippet_json_data("analysis")

        st.session_state["analysis"] = st.text_area(
            "Analysis (editable)",
            value=beautify_text_area(st.session_state.get("analysis")),
            height=700
        )
        # Submit button for saving edits
        save_edits = st.form_submit_button("💾 Save Edits")

        if save_edits:
            st.session_state["saved_flag"] = True
            if st.session_state.get("prompt_suggestions"):
                store_history_snippets_json("prompt_suggestions", st.session_state["prompt_suggestions"])
            store_history_snippets_json("reasoning_prompt", st.session_state["reasoning_prompt"])
            store_history_snippets_json("analysis",st.session_state["analysis"])
            print(f"saved reasoning_prompt and analysis to session_state and latest snippets_{st.session_state.get("run_id")}.json")

    # ✅ Show success/info outside the form so it persists
    if st.session_state.get("saved_flag"):
        st.success(f"✅ Your edited analysis and reasoning prompt have been saved to snippets_{st.session_state.get("run_id")}.json")
        st.info(f"Edits are saved in session state and snippets_{st.session_state.get("run_id")}.json")
        print("Edits are saved in session state.")

    # --- Regenerate code button ---
    st.subheader("Generate Analyzed Code ")
    with st.form("regenerate_form"):
        # Checkbox to decide whether to fetch snippets
        coder_prompt_text = ""

        col1, col2, _ = st.columns([2, 2, 10], gap="xxsmall")
        with col1:
            fetch_snippets_enable = st.checkbox("🔍 Include Faiss snippets", value=True)

        with col2:
            compile_error_check = False
            if not st.session_state.get("compile_error"):
                st.session_state["compile_error"], _, _, _ = get_latest_snippet_json_data("compile_error")
            if st.session_state.get("compile_error"):
                compile_error_check = True
            compile_error_enable = st.checkbox("🔍 fix compile error", value=compile_error_check)
        
        coder_submitted = st.form_submit_button("Generate Code Changes")

        if compile_error_enable:
            coder_prompt_text += f""" based on mql5 source code, header source code, patch files list, after user apply patch files list to mql5 source code, header source code, then compile with metatrader 5 and get compile error. fix compile error.
        """
        else:
            coder_prompt_text += f""" Show MQL5 code changes for function Trade_Strategy."""
        coder_prompt_text += f"""
Return JSON with 4 keys: "old_code", "fix_code", "analysis_code", "explanation_code". 
old_code should show the original function.
fix_code should show the improved function.
analysis_code show the analysis of old_code,
explanation_code show the explanation of fix_code.
        """
        coder_prompt_suggestions = st.text_area("Prompt", coder_prompt_text, height=200)

        if coder_submitted:
            analysis = st.session_state.get("analysis")
            reasoning_prompt = st.session_state.get("reasoning_prompt")

            patch_files_list = {}
            if compile_error_enable:
                patch_files_list["latest_patch"], _, _ = get_latest_snippet_json_data("latest_patch")
                patch_files_list["previous_patch"], _, _ = get_latest_snippet_json_data("previous_patch")

            code_response, reasoning_coder_prompt = suggest_code_improvements(
                OLLAMA_SERVER, 
                reasoning_prompt, 
                analysis, 
                coder_prompt_suggestions, 
                fetch_snippets_enable, 
                compile_error_enable, 
                st.session_state["compile_error"],
                patch_files_list
            )
            st.session_state["coder_prompt"] = coder_prompt_suggestions
            st.session_state["old_code"] = code_response["old_code"]
            st.session_state["fix_code"] = code_response["fix_code"]
            st.session_state["analysis_code"] = code_response["analysis_code"]
            st.session_state["explanation_code"] = code_response["explanation_code"]
            st.session_state["reasoning_coder_prompt"] = reasoning_coder_prompt

        # --- Side-by-side preview ---
        if not st.session_state.get("old_code"):
            get_old_code, _, _, _ = get_latest_snippet_json_data("old_code")
            if get_old_code:
                st.session_state["old_code"] = beautify_text_area(get_old_code)

        if not st.session_state.get("fix_code"):
            get_fix_code, _, _, _ = get_latest_snippet_json_data("fix_code")
            if get_fix_code:
                st.session_state["fix_code"] = beautify_text_area(get_fix_code)
        
        if not st.session_state.get("analysis_code"):
            get_analysis_code, _, _, _ = get_latest_snippet_json_data("analysis_code")
            if get_analysis_code:
                st.session_state["analysis_code"] = beautify_text_area(get_analysis_code)

        if not st.session_state.get("explanation_code"):
            get_explanation_code, _, _, _  = get_latest_snippet_json_data("explanation_code")
            if get_explanation_code:
                st.session_state["explanation_code"] = beautify_text_area(get_explanation_code)

        if st.session_state.get("analysis_code"):
            st.subheader("Code Analysis")
            st.write(st.session_state["analysis_code"])

            st.subheader("Side-by-Side Code Preview")
            if st.session_state.get("old_code") and st.session_state.get("fix_code"):
                col1, col2 = st.columns(2)
                with col1:
                    st.session_state["old_code"] = st.text_area(
                        "old_code",
                        value=st.session_state.get("old_code", ""),
                        height=1000,
                        key="old_code_area"
                    )
                with col2:
                    st.session_state["fix_code"] = st.text_area(
                        "fix_code",
                        value=st.session_state.get("fix_code", ""),
                        height=1000,
                        key="fix_code_area"
                    )
            
            # Optional: show explanation text separately
            if st.session_state.get("explanation_code"):
                st.markdown("**Explanation:**")
                st.write(st.session_state["explanation_code"])


            # --- Download Patch File preview ---
            if st.session_state.get("fix_code"):
                # Submit button for saving edits
                col1, col2, _ = st.columns([1, 2, 10], gap="xxsmall")

                with col1:
                    save_codes = st.form_submit_button("💾 Save Code")

                with col2:
                    generate_patch_submit = st.form_submit_button("🛠️ Generate Patch")

                if save_codes:
                    if st.session_state.get("coder_prompt", {}):
                        store_history_snippets_json("coder_prompt",st.session_state["coder_prompt"] )
                    if st.session_state.get("reasoning_coder_prompt", {}):
                        store_history_snippets_json("reasoning_coder_prompt",st.session_state["reasoning_coder_prompt"] )
                    if st.session_state.get("old_code", {}):
                        store_history_snippets_json("old_code", st.session_state["old_code"])
                    if st.session_state.get("fix_code", {}):
                        store_history_snippets_json("fix_code", st.session_state["fix_code"])
                    if st.session_state.get("explanation_code", {}):
                        store_history_snippets_json("explanation_code", st.session_state["explanation_code"])
                    if st.session_state.get("analysis_code", {}):
                        store_history_snippets_json("analysis_code", st.session_state["analysis_code"])

                if generate_patch_submit:
                    try:
                        patch_path = generate_patch_from_git(
                            os.getenv("EA_CODE_GIT_REPO"),
                            st.session_state["fix_code"],
                            os.getenv("EA_MQ5_SUBPATH"),
                            os.getenv("EA_HEADER_SUBPATH"),
                            os.getenv("AGENT_PATH")
                        )

                        st.success(f"Patch file generated: {patch_path}")
                        # get the 
                        st.session_state["latest_patch_path"] = patch_path
                        st.session_state["latest_patch"] = get_patch_content(patch_path)
                        if st.session_state.get("latest_patch_path") and st.session_state.get("latest_patch"):
                            # check memory.json RUN_HISTORY run_id, check if their /logs/{run_id}/snippet_{run_id}.json
                            # if that particular snippet_{run_id}.json contain history previous_patch and latest_patch, put to current 
                            
                            get_previous_patch_json, _, _, _ = get_latest_snippet_json_data("previous_patch", False)
                            get_latest_patch_json, archieve, run_id, run_id_num = get_latest_snippet_json_data("latest_patch", False)
                            
                            if get_latest_patch_json and isinstance(get_latest_patch_json, dict):
                                new_prev_patch_run_id = get_latest_patch_json.get("extras", {}).get("run_id", run_id)
                                new_prev_patch_run_id_num = get_latest_patch_json.get("extras", {}).get("run_id_num", run_id_num)
                                print(f"found extras in get_previous_patch_json")

                                # update previous_patch only if run_id/run_id_num differ
                                if get_previous_patch_json:
                                    print(f"prev get_latest_patch_json info {get_previous_patch_json["extras"]["run_id"]} and {get_latest_patch_json["extras"]["run_id_num"]}")
                                    if (get_previous_patch_json["extras"]["run_id"] != run_id or
                                        get_previous_patch_json["extras"]["run_id_num"] != run_id_num):
                                        if run_id not in get_previous_patch_json:
                                            get_previous_patch_json[run_id] = {}
                                        get_previous_patch_json[new_prev_patch_run_id][new_prev_patch_run_id_num] = get_latest_patch_json
                                        store_history_snippets_json("previous_patch", get_previous_patch_json, None, False)
                                        print("Successful update latest_patch to previous_patch")
                                else:
                                    if (get_latest_patch_json["extras"]["run_id"] != run_id or
                                        get_latest_patch_json["extras"]["run_id_num"] != run_id_num):
                                        new_prev_patch = {
                                            new_prev_patch_run_id: {
                                                new_prev_patch_run_id_num: get_latest_patch_json
                                            }
                                        }
                                        store_history_snippets_json("previous_patch", new_prev_patch, None, False)
                                        print("Successful create previous_patch")

                            # ✅ Build latest_patch dict in the desired format
                            patch_content = st.session_state["latest_patch"]
                            st.session_state["latest_patch"] = {
                                "chunks": [patch_content],  # wrap string in list
                                "token": count_tokens(patch_content, os.getenv("CODER_AGENT"), False),
                                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "extras": {
                                    "path": st.session_state["latest_patch_path"],
                                    "run_id": run_id,
                                    "run_id_num": run_id_num,
                                }
                            }
                            store_history_snippets_json(
                                "latest_patch",
                                st.session_state["latest_patch"],
                                False
                            )
                            st.info("You can now apply it using the Apply button below.")
                    except RuntimeError as e:
                        st.error(str(e))
                
            if st.session_state.get("latest_patch_path"):
                patch_submit = st.form_submit_button("💾 Apply Patch")
                github_push_enable = st.checkbox("🔍 Apply to Github")
                if patch_submit:
                    full_commit_message = apply_patch_to_git(
                        repo_dir=os.getenv("EA_CODE_GIT_REPO"),
                        patch_file=st.session_state["latest_patch_path"],
                        github_push=github_push_enable,
                    )
                    if full_commit_message:
                        store_history_snippets_json("patch_to_git", f"{full_commit_message} push={github_push_enable}")
                    st.success(f"Applied patch file {st.session_state["latest_patch_path"]} to git repo {os.getenv("EA_CODE_GIT_REPO")} with commit {full_commit_message}")

if __name__ == "__main__":
    # Example old vs new code
    st.set_page_config(layout="wide")
    improvements = "Added type safety to add() function."
    commit_message = "Enhance add() with type safety"

    preview_improvements()


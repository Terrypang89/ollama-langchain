import streamlit as st
import subprocess
import difflib
import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM

load_dotenv()
OLLAMA_SERVER = os.getenv("OLLAMA_API_BASE")
ollama_model = OllamaLLM(model="mistral:7b", base_url=OLLAMA_SERVER)

def preview_improvements(old_code_file, code_repo, improvements, commit_message):
    st.title("Backtest Improvements Review")

    # --- Load old code from file ---
    if not os.path.exists(old_code_file):
        st.error(f"File not found: {old_code_file}")
        return
    with open(old_code_file, "r", encoding="utf-8") as f:
        old_code = f.read()

    # --- Ensure repo is initialized ---
    if not os.path.exists(os.path.join(code_repo, ".git")):
        try:
            subprocess.run(["git", "init"], cwd=code_repo, check=True)
            # Add .gitignore to exclude patch files
            gitignore_path = os.path.join(code_repo, ".gitignore")
            if not os.path.exists(gitignore_path):
                with open(gitignore_path, "w", encoding="utf-8") as gi:
                    gi.write("*.patches\n")
            subprocess.run(["git", "add", "."], cwd=code_repo, check=True)
            subprocess.run(["git", "commit", "-m", "init with files " + old_code_file ], cwd=code_repo, check=True)
            st.info("Initialized new Git repository and staged files.")
        except subprocess.CalledProcessError as e:
            st.error(f"Git init/add failed: {e}")
            return

    # --- Improvements summary ---
    st.subheader("Proposed Improvements Summary")
    st.text_area("Improvements", improvements, height=200)

    # --- Commit message editor ---
    st.subheader("Commit Message")
    user_commit_message = st.text_area("Commit Message", commit_message, height=100)

    # --- Suggestions box ---
    st.subheader("Suggest Code Changes")
    user_suggestions = st.text_area("Your Suggestions", "", height=200)

    # --- Regenerate button ---
    if st.button("Regenerate Code Changes"):
        if user_suggestions.strip() or improvements.strip():
            prompt = f"""Here is the original code:\n{old_code}\n
Apply these improvements:\n{improvements}\n
And also consider these suggestions:\n{user_suggestions}\n
Please output the full updated code without commented:"""

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

    # --- Download patch ---
    effective_new_code = st.session_state.get("new_code", old_code)
    diff = list(difflib.unified_diff(
        old_code.splitlines(),
        effective_new_code.splitlines(),
        fromfile=os.path.basename(old_code_file),   # "Testcode.py"
        tofile=os.path.basename(old_code_file),     # "Testcode.py"
        lineterm=""
    ))

    # patch_text = "\n".join(diff)
    if "new_code" in st.session_state:
        if st.button("Download Patch File"):
            try:
                # Overwrite file with regenerated code
                with open(old_code_file, "w", encoding="utf-8") as f:
                    f.write(st.session_state["new_code"])

                # Stage and commit
                # subprocess.run(["git", "add", old_code_file], cwd=code_repo, check=True)
                subprocess.run(["git", "add", os.path.basename(old_code_file)], cwd=code_repo, check=True)

                subprocess.run(["git", "commit", "-m", user_commit_message], cwd=code_repo, check=True)
                
                patch_dir = os.path.join(code_repo, "patches")
                if not os.path.exists(patch_dir):
                    os.makedirs(patch_dir)
                # subprocess.run(["git", "format-patch", "-1", "HEAD", "-o", "patches"], cwd=code_repo, check=True)
                result = subprocess.run(
                    ["git", "format-patch", "-1", "HEAD", "-o", "patches"],
                    cwd=code_repo,
                    check=True,
                    capture_output=True,
                    text=True
                )

                # The stdout contains the filename, e.g. "0001-your-commit-message.patch\n"
                patch_filename = result.stdout.strip()
                patch_path = os.path.join(patch_dir, patch_filename)

                # Revert back to baseline so patch can be applied later
                subprocess.run(["git", "reset", "--hard", "HEAD~1"], cwd=code_repo, check=True)

                # st.success("Patch file generated with git format-patch and repo reset to baseline!")
                st.success(f"Patch file generated: {patch_path}")
                st.session_state["latest_patch_path"] = patch_path
                st.info("You can now apply it using the Apply button below.")
            except subprocess.CalledProcessError as e:
                st.error(f"Git error while generating patch: {e}")


    # --- Apply button (optional hunks approval) ---
    st.subheader("Code Diff Hunks (Approve Individually)")
    approved_hunks = []
    current_hunk = []

    for line in diff:
        if line.startswith("@@"):
            if current_hunk:
                hunk_text = "\n".join(current_hunk)
                approve = st.checkbox(f"Approve hunk starting {current_hunk[0]}", value=True)
                if approve:
                    approved_hunks.append(hunk_text)
                current_hunk = []
            current_hunk.append(line)
        else:
            current_hunk.append(line)

    if current_hunk:
        hunk_text = "\n".join(current_hunk)
        approve = st.checkbox(f"Approve hunk starting {current_hunk[0]}", value=True)
        if approve:
            approved_hunks.append(hunk_text)

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


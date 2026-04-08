import os
import shutil
from testMT5 import get_last_run_info

def clean_store(store_path):
    """
    Clean the backtest store environment:
    - Delete files_index folder
    - Delete logs folder
    - Delete MQL5 folder (force remove, even if it has .git)
    - Delete memory.json file
    """
    targets = [
        os.path.join(store_path, "files_index"),
        os.path.join(store_path, "logs"),
        os.path.join(store_path, "MQL5"),
        os.path.join(store_path, "memory.json"),
    ]

    for target in targets:
        try:
            if os.path.isdir(target):
                # force remove read-only files
                def on_rm_error(func, path, exc_info):
                    import stat
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                shutil.rmtree(target, onerror=on_rm_error)
                print(f"Deleted folder: {target}")
            elif os.path.isfile(target):
                os.remove(target)
                print(f"Deleted file: {target}")
            else:
                print(f"Not found, skipped: {target}")
        except Exception as e:
            print(f"Error deleting {target}: {e}")

def clean_snippets_json():
    info = get_last_run_info()
    run_id = info["run_id"]
    archieve_dir = info["archieve"]   # consistent spelling
    
    # Check archieve directory
    if not archieve_dir or not os.path.exists(archieve_dir):
        raise FileNotFoundError(f"❌ Archive directory not found: {archieve_dir}")

    archieve_file = os.path.join(archieve_dir, f"snippets_{run_id}.json")

    if not os.path.exists(archieve_file):
        raise FileNotFoundError(f"❌ Archive file not found for run_id {run_id}: {archieve_file}")
    else:
        os.remove(archieve_file)
        print(f"Deleted file: {archieve_file}")


# clean_store("./")

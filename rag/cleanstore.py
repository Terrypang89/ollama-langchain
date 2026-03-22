import os
import shutil

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


clean_store("./")

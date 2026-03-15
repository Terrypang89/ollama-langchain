import subprocess
import os
from pathlib import Path
import MetaTrader5 as mt5
from dotenv import load_dotenv
import time
import psutil  # install with pip if needed

load_dotenv()

def compile_ea(mq5_file, metaeditor_path):
    """
    Compile an EA using MetaEditor.
    mq5_file: path to your .mq5 file
    metaeditor_path: path to MetaEditor.exe (usually in MT5 installation folder)
    """
    mq5_file = Path(mq5_file).resolve()
    cmd = [
        metaeditor_path,
        "/compile:" + str(mq5_file),
        "/log"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print("Compiler stdout:", result.stdout)
    print("Compiler stderr:", result.stderr)

    # Check MetaEditor log file
    log_dir = Path(metaeditor_path).parent / "logs"
    log_file = log_dir / "MetaEditor.log"
    if log_file.exists():
        print("MetaEditor log:")
        print(log_file.read_text(encoding="utf-8", errors="ignore"))

    return result.returncode == 0


def run_mt5_backtest(config_path, terminal_path, portable_enable=True, timeout=60):
    """
    Run MT5 backtest via terminal command and return logs.
    """
    # Build the command
    if portable_enable:
        cmd = [terminal_path, "/portable", f"/config:{config_path}"]
    else:
        cmd = [terminal_path, f"/config:{config_path}"]

    # Start process
    proc = subprocess.Popen(cmd)

    try:
        # Wait for completion (timeout in seconds)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print("Backtest timed out, killing process...")
        proc.kill()

    return proc.returncode

def parse_backtest_report(report_file):
    df = pd.read_html(report_file)[0]  # if HTML report
    return df

# EA_MQL_FILE = os.getenv("EA_MQL_FILE")
# METAEDITOR_PATH = os.getenv("METAEDITOR_PATH")
# result = compile_ea(EA_MQL_FILE, METAEDITOR_PATH)

TERMINAL_PATH = os.getenv("TERMINAL_PATH")
print("TERMINAL_PATH:", TERMINAL_PATH)
EA_EX_FILE = os.getenv("EA_EX_FILE")
print("EA_EX_FILE:", EA_EX_FILE)
COMPILE_LOG_FILE = os.getenv("COMPILE_LOG_FILE")
print("COMPILE_LOG_FILE:", COMPILE_LOG_FILE)
EA_INI_PATH = os.getenv("EA_INI_PATH")
result = run_mt5_backtest(EA_INI_PATH, TERMINAL_PATH, False, 20)
# Print("run_mt5_backtest:", run_mt5_backtest)
# run_python_mt5_backtest(EA_EX_FILE, "EURUSD", "H1")
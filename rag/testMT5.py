import subprocess
import os
from pathlib import Path
import MetaTrader5 as mt5
from dotenv import load_dotenv
import pandas as pd
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

def run_mt5_backtest(config_path, terminal_path, report_path, log_path, portable_enable=True, timeout=60):
    """
    Run MT5 backtest via terminal command with timeout.
    Return the report file path if generated, else logs.
    """
    # Build command
    if portable_enable:
        cmd = [terminal_path, "/portable", f"/config:{config_path}"]
    else:
        cmd = [terminal_path, f"/config:{config_path}"]

    # Start process
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        # Wait for completion (timeout in seconds)
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        print("Backtest timed out, killing process...")
        proc.kill()
        stdout, stderr = proc.communicate()

    # Extract report path from ini file
    report_file = None
    with open(config_path, "r") as f:
        for line in f:
            if line.strip().startswith("Report="):
                report_file = line.strip().split("=", 1)[1]
                break

    # Normalize path (if relative, make absolute relative to ini folder)
    if report_file:
        if not os.path.isabs(report_file):
            report_file = os.path.join(report_path, report_file)
            print("report_file2:", report_file)

     # Find latest log file
    latest_log_file = None
    if os.path.exists(log_path):
        log_files = [os.path.join(log_path, f) for f in os.listdir(log_path)
                     if os.path.isfile(os.path.join(log_path, f))]
        if log_files:
            latest_log_file = max(log_files, key=os.path.getmtime)

    if os.path.exists(report_file) and os.path.exists(latest_log_file):
        return report_file, latest_log_file

    # If report not found, return logs instead
    return stdout, stderr

def parse_backtest_report(report_file):
    """
    Parse MT5 backtest HTML report into a DataFrame.
    """
    if not report_file or not os.path.exists(report_file):
        raise FileNotFoundError(f"Report file not found or empty: {report_file}")

    tables = pd.read_html(report_file)
    if not tables:
        raise ValueError("Report file is empty or contains no tables")

    return tables[0]  

# EA_MQL_FILE = os.getenv("EA_MQL_FILE")
# METAEDITOR_PATH = os.getenv("METAEDITOR_PATH")
# result = compile_ea(EA_MQL_FILE, METAEDITOR_PATH)

ini_file = update_ini_file(
    ini_path="tester.ini",
    login=os.getenv("MT5_LOGIN"),
    password=os.getenv("MT5_PASSWORD"),
    server=os.getenv("MT5_SERVER"),
    expert="Tofu_EA_Pro.ex5",
    symbol="XAUUSD",
    period="M5",
    from_date="2025.04.01",
    to_date="2025.04.05",
    report_path=r"Tester_report.html",
    updates={
        "Expert": "Tofu_EA_Pro.ex5",
        "Symbol": "XAUUSD",
        "Period": "M5",
        "FromDate": "2025.05.01",
        "ToDate": "2025.05.10",
        "Visual" : False,
        "Report" : "Tester_report.html",
    }
)

TERMINAL_PATH = os.getenv("TERMINAL_PATH")
print("TERMINAL_PATH:", TERMINAL_PATH)
EA_EX_FILE = os.getenv("EA_EX_FILE")
print("EA_EX_FILE:", EA_EX_FILE)
COMPILE_LOG_FILE = os.getenv("COMPILE_LOG_FILE")
print("COMPILE_LOG_FILE:", COMPILE_LOG_FILE)
# EA_INI_PATH = os.getenv("EA_INI_PATH")
BACKTEST_REPORT_PATH = os.getenv("BACKTEST_REPORT_PATH")
BACKTEST_LOG_PATH = os.getenv("BACKTEST_LOG_PATH")
print("BACKTEST_REPORT_PATH:", BACKTEST_REPORT_PATH)
print("BACKTEST_LOG_PATH:", BACKTEST_LOG_PATH)

report_file, log_file = run_mt5_backtest(ini_file, TERMINAL_PATH, BACKTEST_REPORT_PATH, BACKTEST_LOG_PATH, False, 30)
print("report_file:", report_file)
print("log_file:", log_file)
df = parse_backtest_report(report_file)
print(df.head())
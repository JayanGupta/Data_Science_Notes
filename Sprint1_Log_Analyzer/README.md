# Sprint 1: Automated Data Pipeline & Forensic Log Analyzer

## Executive Overview

This project simulates a production ingestion and anomaly-detection pipeline in an enterprise telemetry environment. The system processes high-volume server logs arriving from distributed microservices, filters non-data junk files, safely parses records using Object-Oriented Programming (OOP) with targeted exception handling, tallies critical infrastructure errors (`5xx`), and separates valid records from corrupted files.

---

## Data Architecture & File Placement

### 1. Data Source & Download Location
If you are initializing this project from scratch or need to restore the pristine dataset, all required raw files are located in the repository's centralized datasets hub:
- **Source Path:** `datasets/Sprint_1_Log_Analyzer/daily_logs/`
- **Target Working Path:** `Sprint1_Log_Analyzer/data/daily_logs/`

### 2. Dataset Composition (75 Total Files)
- **40 Valid JSON Files:** Structured dictionary logs containing `ip_address`, `status_code`, and `timestamp`.
- **25 Valid TXT Files:** Comma-separated logs formatted as `ip_address,status_code`.
- **7 Intentionally Corrupted Files:**
  - 2 malformed JSON syntax errors (`json.JSONDecodeError`)
  - 3 missing required schema keys (`KeyError` on `ip_address` or `status_code`)
  - 2 malformed TXT records missing the comma delimiter (`ValueError`)
- **3 Hidden / Junk Files:** Non-data files (`.DS_Store`, `~backup_event_014.json`, `notes.md`) designed to test directory filtering logic.

---

## Directory Structure

### Expected Structure Before Execution
```text
Sprint1_Log_Analyzer/
├── README.md                          # Project documentation and operational guide
├── Sprint1_Solution.ipynb             # Executed solution notebook in Q&A format
└── data/
    └── daily_logs/                    # 75 raw input log files (.txt and .json)
```

### Expected Structure After Execution
```text
Sprint1_Log_Analyzer/
├── README.md
├── Sprint1_Solution.ipynb
├── analytics_ready.json               # [OUTPUT] 65 validated clean records
└── data/
    ├── daily_logs/                    # 75 raw input log files (unmodified)
    └── quarantine/                    # [OUTPUT] 7 isolated corrupted files
```

---

## How to Run the Project

### Option A: Interactive In-Class Coding (Jupyter Notebook)
Launch the solution notebook to follow or code the pipeline live:
```bash
cd Sprint1_Log_Analyzer
jupyter notebook Sprint1_Solution.ipynb
```

### Option B: Headless Batch Execution
Execute the entire pipeline non-interactively via `nbconvert`:
```bash
cd Sprint1_Log_Analyzer
jupyter nbconvert --to notebook --execute --inplace Sprint1_Solution.ipynb
```

---

## Verification & Output Audit

Following a successful pipeline run, verify data governance metrics:
1. **Output File:** `analytics_ready.json` must contain exactly **65** formatted records.
2. **Quarantine Directory:** `data/quarantine/` must contain exactly **7** corrupted files.
3. **Forensic Aggregation:** The top offending IP address triggering `5xx` server errors must be identified and tallied.

### Quick Verification Command (PowerShell)
```powershell
python -c "import json, os; assert len(json.load(open('analytics_ready.json'))) == 65; assert len(os.listdir('data/quarantine')) == 7; print('[PASS] All Sprint 1 verification assertions passed!')"
```

---

## Environment Reset for Live Demonstrations

To demonstrate the pipeline creating `analytics_ready.json` and `data/quarantine/` from scratch in front of students:
```powershell
# Remove previously generated outputs
Remove-Item -Path "analytics_ready.json" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "data/quarantine" -Recurse -Force -ErrorAction SilentlyContinue

# Verify data folder is pristine
Get-ChildItem -Path "data/daily_logs" | Measure-Object | Select-Object -ExpandProperty Count
# Expected count: 75 files
```

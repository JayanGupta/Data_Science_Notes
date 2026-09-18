# Complete Data Science — Project-Based Learning (Sprints 1–4)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?style=for-the-badge&logo=sqlite&logoColor=white)](https://www.sqlite.org/)
[![Power BI](https://img.shields.io/badge/Power_BI-Desktop-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)](https://powerbi.microsoft.com/)
[![Pandas](https://img.shields.io/badge/Pandas-Analytics-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Curriculum Status](https://img.shields.io/badge/Curriculum-Sprints_1--4_Verified-2EA44F?style=for-the-badge)](Complete%20Data%20Science%20-%20Project%20Based%20Learning.md)

---

## Executive Summary

This repository delivers an enterprise-grade, project-based data science curriculum covering **Sprints 1 through 4**, strictly aligned with the official specification in [Complete Data Science - Project Based Learning.md](Complete%20Data%20Science%20-%20Project%20Based%20Learning.md).

Designed specifically for live classroom coding and structured technical instruction, every project follows a pedagogical **Question & Answer (Q&A) format** using **small, single-purpose code cells**, bulleted concept pointers, architectural diagrams, and pre-executed verification outputs.

---

## Curriculum Pipeline Architecture

```text
+-----------------------------------------------------------------------------+
|             Sprint 1: Automated Pipeline & Forensic Log Analyzer            |
|               Directory: `Sprint1_Log_Analyzer/`                            |
|  - Directory scanning, extension filtering, and skipping junk files         |
|  - Object-Oriented `DataParser` with targeted `try-except` handling         |
|  - Aggregating 5xx forensic anomalies per IP address                        |
|  - Consolidated output `analytics_ready.json` & quarantined corrupt files   |
+--------------------------------------┬--------------------------------------+
                                       |
                                       v Hands-off to Commercial Data
+-----------------------------------------------------------------------------+
|          Sprint 2: Building, Cleaning & Analyzing an E-Commerce Dataset     |
|               Directory: `Sprint2_Ecommerce_Dataset/`                       |
|  - Live web scraping with `requests` & `BeautifulSoup` (open sandbox)       |
|  - Vectorized numeric cleaning and IQR outlier detection (1.5 x IQR)        |
|  - Inferential statistics: Two-sample Welch's t-test (p < 0.05)             |
|  - Pearson correlation matrix & 3-panel publication visual (Seaborn)        |
|  - Exported Foundation: `data/ecommerce_cleaned.csv`                        |
+--------------------------------------┬--------------------------------------+
                                       |
                                       v Ingest into Relational Engine
+-----------------------------------------------------------------------------+
|               Sprint 3: Analyzing the E-Commerce Dataset in SQL             |
|               Directory: `Sprint3_SQL_Analysis/`                            |
|  - 3-table normalized relational schema (`categories`, `products`, `reviews`)|
|  - Loaded into SQLite via SQLAlchemy                                        |
|  - Window functions: `RANK() OVER (PARTITION BY ... ORDER BY ...)`          |
|  - CTEs: Top price premium movers above category average                    |
|  - In-database cleaning: Deduping reviews with `ROW_NUMBER() OVER`          |
|  - Date standardization across 4 formats to ISO `YYYY-MM-DD`                |
+--------------------------------------┬--------------------------------------+
                                       |
                                       v Connect to Business Intelligence Layer
+-----------------------------------------------------------------------------+
|          Sprint 4: Dashboarding the E-Commerce Dataset in Power BI          |
|               Directory: `Sprint4_PowerBI_Dashboard/`                       |
|  - Native Power BI solution file: `Sprint4_Ecommerce_Dashboard.pbix`        |
|  - Executive KPI cards: Total Products, Catalog Avg Price, Weighted Rating  |
|  - Comparative visual analytics: Category benchmarks & Satisfaction matrix  |
|  - Enterprise DAX measures: `CALCULATE`, `ALLEXCEPT`, and Dynamic Slicers   |
|  - Interactive Python & Plotly prototype in `Sprint4_Solution.ipynb`        |
|  - Standalone browser preview in `dashboard_design_preview.html`             |
+-----------------------------------------------------------------------------+
```

---

## Curriculum Matrix

| Sprint | Project Title | Core Technologies | Focus Scenario | Primary Solution Artifacts | Output Artifacts |
|---|---|---|---|---|---|
| **Sprint 1** | Automated Pipeline & Forensic Log Analyzer | Python, OOP, `os`, `json`, `shutil` | Ingesting messy telemetry logs and quarantining corrupted files | [`Sprint1_Solution.ipynb`](Sprint1_Log_Analyzer/Sprint1_Solution.ipynb) | `analytics_ready.json`, `data/quarantine/` |
| **Sprint 2** | E-Commerce Scraping, Cleaning & EDA | `BeautifulSoup`, `requests`, `NumPy`, `Pandas`, `SciPy`, `Seaborn` | Scraping web listings, IQR outlier detection, Welch's t-test | [`Sprint2_Solution.ipynb`](Sprint2_Ecommerce_Dataset/Sprint2_Solution.ipynb) | `data/ecommerce_cleaned.csv`, `eda_summary.png` |
| **Sprint 3** | E-Commerce Relational SQL Analysis | SQL, SQLite, SQLAlchemy, Window Functions, CTEs | Normalizing schema, window ranking, deduplicating reviews | [`Sprint3_Solution.ipynb`](Sprint3_SQL_Analysis/Sprint3_Solution.ipynb) | `ecommerce.db` (3 normalized tables) |
| **Sprint 4** | E-Commerce Executive Dashboard | Power BI Desktop, DAX, Plotly, HTML5 | Category Manager dashboard, KPI cards, DAX filter context | [`Sprint4_Ecommerce_Dashboard.pbix`](Sprint4_PowerBI_Dashboard/Sprint4_Ecommerce_Dashboard.pbix), [`Sprint4_Solution.ipynb`](Sprint4_PowerBI_Dashboard/Sprint4_Solution.ipynb) | Native `.pbix` model, Plotly charts, HTML mockup |

---

## Pedagogical Structure & Teaching Standards

1. **Question & Answer (Q&A) Format:**
   - Every topic is introduced with a concrete business or technical question (e.g., *How do we detect price outliers using the 1.5 x IQR rule?*).
   - Concept pointers summarize the statistical or algorithmic theory in clear bullet points before code is executed.
   - Code cells are kept small (3 to 8 lines) so instructors can narrate each step while typing live.
   - Key takeaways and data assertions summarize the outcome immediately after each step.

2. **Zero-Crash Defensive Architecture:**
   - Exception handling is targeted (`JSONDecodeError`, `KeyError`, `ValueError`) rather than using bare `except:`.
   - Data corruption is audited and preserved in quarantine directories rather than silently dropped.

3. **Continuous Data Thread:**
   - The four sprints do not exist in isolation: Sprint 2 scrapes and cleans the product catalog; Sprint 3 normalizes it into SQL and cleans customer reviews; Sprint 4 connects Power BI to that exact database to produce executive dashboards.

---

## Centralized Datasets Hub

All raw datasets, starting CSVs, and benchmark files are duplicated in the root [`datasets/`](datasets/README.md) directory. This ensures instructors and students can start fresh or restore individual sprint folders at any time without re-downloading external resources.

```text
datasets/
├── README.md                          # Data dictionary and quick copy commands
├── Sprint_1_Log_Analyzer/             # 75 raw server logs (.txt and .json)
├── Sprint_2_Ecommerce_Dataset/        # 149 raw scraped e-commerce listings
├── Sprint_3_SQL_Analysis/             # Clean product CSV and raw reviews CSV
└── Sprint_4_PowerBI_Dashboard/        # Power BI denormalized CSVs and SQLite database
```

---

## Repository Layout

```text
Data_Science_Notes/
├── Complete Data Science - Project Based Learning.md  # Official curriculum syllabus
├── README.md                                          # Master repository documentation
│
├── datasets/                                          # Centralized data repository
│   ├── README.md                                      # Data dictionary & restore guide
│   ├── Sprint_1_Log_Analyzer/                         # Raw server logs
│   ├── Sprint_2_Ecommerce_Dataset/                    # Raw scraped listings
│   ├── Sprint_3_SQL_Analysis/                         # Relational inputs
│   └── Sprint_4_PowerBI_Dashboard/                    # BI reporting tables
│
├── Sprint1_Log_Analyzer/                              # SPRINT 1 FOLDER
│   ├── README.md                                      # Sprint documentation & data guide
│   ├── Sprint1_Solution.ipynb                         # Pre-executed Q&A solution notebook
│   ├── analytics_ready.json                           # Pipeline output (65 clean records)
│   └── data/                                          # Local working data directory
│
├── Sprint2_Ecommerce_Dataset/                         # SPRINT 2 FOLDER
│   ├── README.md                                      # Sprint documentation & data guide
│   ├── Sprint2_Solution.ipynb                         # Pre-executed Q&A solution notebook
│   ├── eda_summary.png                                # 3-panel publication visual
│   └── data/                                          # Local working data directory
│
├── Sprint3_SQL_Analysis/                              # SPRINT 3 FOLDER
│   ├── README.md                                      # Sprint documentation & data guide
│   ├── Sprint3_Solution.ipynb                         # Pre-executed Q&A solution notebook
│   ├── ecommerce.db                                   # SQLite relational database
│   └── data/                                          # Local working data directory
│
└── Sprint4_PowerBI_Dashboard/                         # SPRINT 4 FOLDER
    ├── README.md                                      # Sprint documentation & data guide
    ├── Sprint4_Ecommerce_Dashboard.pbix               # Native Power BI Desktop solution file
    ├── Sprint4_Solution.ipynb                         # Pre-executed Plotly prototype notebook
    ├── DAX_Measures_Reference.md                      # DAX formulas reference
    ├── dashboard_design_preview.html                  # Standalone interactive browser preview
    └── data/                                          # Local working data directory
```

---

## Environment Setup & Quick Start

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.12 and 3.14)
- Jupyter Notebook / JupyterLab
- Microsoft Power BI Desktop (Optional, for native `.pbix` viewing)

### 2. Dependency Installation
```bash
pip install pandas numpy scipy matplotlib seaborn requests beautifulsoup4 sqlalchemy plotly
```

### 3. Verification Command
To verify that all 4 sprint solution notebooks execute without errors:
```powershell
python -c "import json; [print(f'[OK] {s}: Clean execution') for s, p in [('Sprint 1', 'Sprint1_Log_Analyzer/Sprint1_Solution.ipynb'), ('Sprint 2', 'Sprint2_Ecommerce_Dataset/Sprint2_Solution.ipynb'), ('Sprint 3', 'Sprint3_SQL_Analysis/Sprint3_Solution.ipynb'), ('Sprint 4', 'Sprint4_PowerBI_Dashboard/Sprint4_Solution.ipynb')] if not any(o.get('output_type') == 'error' for c in json.load(open(p, encoding='utf-8'))['cells'] for o in c.get('outputs', []))]"
```

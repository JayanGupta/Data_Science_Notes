<div align="center">

<!-- Animated Typing Header -->
<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=34&pause=1000&color=00D9FF&center=true&vCenter=true&multiline=true&width=900&height=95&lines=Data+Science+Project-Based+Learning;Sprints+1+to+4+Live+Industry+Curriculum" alt="Typing SVG" /></a>

<br/>

<!-- Animated Wave Banner -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:1a1b27,100:00d9ff&height=220&section=header&text=Data%20Science%20Notes&fontSize=48&fontColor=ffffff&animation=fadeIn&fontAlignY=35&desc=Production-Grade%20Classroom%20Curriculum%20%7C%20Sprints%201%20to%204&descSize=18&descAlignY=55&descColor=58a6ff" width="100%"/>

<br/>

<!-- Badges Row 1: Repository Status -->
![GitHub last commit](https://img.shields.io/github/last-commit/JayanGupta/Data_Science_Notes?style=for-the-badge&color=00d9ff&labelColor=0d1117)
![GitHub repo size](https://img.shields.io/github/repo-size/JayanGupta/Data_Science_Notes?style=for-the-badge&color=7c3aed&labelColor=0d1117)
![GitHub stars](https://img.shields.io/github/stars/JayanGupta/Data_Science_Notes?style=for-the-badge&color=fbbf24&labelColor=0d1117)
![GitHub forks](https://img.shields.io/github/forks/JayanGupta/Data_Science_Notes?style=for-the-badge&color=34d399&labelColor=0d1117)

<br/>

<!-- Badges Row 2: Core Data Technologies -->
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![Power BI](https://img.shields.io/badge/Power_BI-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

<br/>

<!-- Badges Row 3: Developer & Teaching Tooling -->
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![BeautifulSoup](https://img.shields.io/badge/Beautiful_Soup-59666C?style=for-the-badge&logo=python&logoColor=white)
![Git](https://img.shields.io/badge/Git-F05032?style=for-the-badge&logo=git&logoColor=white)
![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)

<br/><br/>

<!-- Profile Views Counter -->
![Profile Views](https://komarev.com/ghpvc/?username=JayanGupta&label=Repository+Views&color=00d9ff&style=for-the-badge)

</div>

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## About This Repository

> *A complete, live-coding project curriculum engineered for Data Science & Engineering cohorts — spanning server log telemetry parsing, live web scraping with EDA, relational SQL modeling, and interactive Power BI executive dashboards.*

This repository acts as a **ready-to-teach, live-coding classroom suite**. Rather than abstract theory or toy datasets, every sprint implements a real-world software pipeline solving genuine production data challenges.

### Core Curriculum Highlights
* **Industry Pipeline Continuity**: Output from Sprint 2 cleans raw listings and feeds Sprint 3's relational database, which directly populates Sprint 4's executive dashboard.
* **Bite-Sized Pedagogical Code Cells**: Code is split into focused 3-to-8 line single-action cells ideal for step-by-step in-class live coding.
* **Question-and-Answer Framework**: Every phase starts with an educational inquiry, concept pointers before the code, and structured takeaway notes after execution.
* **Centralized Datasets Hub**: All benchmark inputs and raw data files are preserved in a dedicated `datasets/` root folder for instant download and demonstration resets.
* **Native Power BI Solution**: Includes an end-to-end `.pbix` solution file alongside Python Plotly interactive dashboard mirrors and complete DAX measure formulas.

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## End-to-End Curriculum Pipeline

```mermaid
graph TD
    subgraph Sprint1["Sprint 1: Log Analyzer (Python OOP)"]
        S1_In["74 Mixed Server Logs<br/>(JSON & TXT)"] --> S1_Parse["DataParser OOP Pipeline<br/>Regex & JSON Validator"]
        S1_Parse --> S1_Out1["analytics_ready.json<br/>(65 Valid Records)"]
        S1_Parse --> S1_Out2["quarantine/<br/>(7 Corrupted Files)"]
    end

    subgraph Sprint2["Sprint 2: E-Commerce Dataset (Scraping & EDA)"]
        S2_Scrape["Live HTTP Scraper<br/>(requests + BeautifulSoup)"] --> S2_Raw["ecommerce_raw_scraped.csv<br/>(149 Raw Listings)"]
        S2_Raw --> S2_Clean["Vectorized Cleaning & IQR<br/>T-Test (p < 0.05) & Pearson Corr"]
        S2_Clean --> S2_Out["ecommerce_cleaned.csv<br/>(103 Products)"]
    end

    subgraph Sprint3["Sprint 3: SQL Analytics (SQLite & SQLAlchemy)"]
        S3_In1["ecommerce_cleaned.csv<br/>(Sprint 2 Products)"] --> S3_DB[("ecommerce.db<br/>Normalized 3NF")]
        S3_In2["reviews_raw.csv<br/>(595 Customer Reviews)"] --> S3_DB
        S3_DB --> S3_Window["Window Functions & CTEs<br/>ROW_NUMBER Deduplication"]
        S3_Window --> S3_CleanReviews["574 Cleaned Reviews<br/>ISO Standardized Dates"]
    end

    subgraph Sprint4["Sprint 4: Executive BI Dashboard (Power BI Desktop)"]
        S4_Data1["products_for_powerbi.csv"] --> S4_PBI["Sprint4_Ecommerce_Dashboard.pbix<br/>Executive KPI Canvas"]
        S4_Data2["review_summary_for_powerbi.csv"] --> S4_PBI
        S4_PBI --> S4_DAX["DAX Measures: Weighted Ratings,<br/>Discount Depth & Rating Badges"]
        S4_PBI --> S4_Web["dashboard_design_preview.html<br/>Plotly Interactive Prototype"]
    end

    S2_Out -. "Feeds Relational Products" .-> S3_In1
    S3_DB -. "Feeds BI Data Model" .-> S4_Data1
    S3_CleanReviews -. "Feeds Review Metrics" .-> S4_Data2

    style Sprint1 fill:#0d1117,stroke:#00d9ff,stroke-width:2px,color:#ffffff
    style Sprint2 fill:#0d1117,stroke:#34d399,stroke-width:2px,color:#ffffff
    style Sprint3 fill:#0d1117,stroke:#7c3aed,stroke-width:2px,color:#ffffff
    style Sprint4 fill:#0d1117,stroke:#fbbf24,stroke-width:2px,color:#ffffff
```

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## Curriculum Sprint Suite

| Sprint | Domain & Focus | Real-World Challenge | Technologies | Key Artifacts |
| :--- | :--- | :--- | :--- | :--- |
| **[Sprint 1](Sprint1_Log_Analyzer/)** | **Log Analyzer** | Ingest 74 chaotic server logs across mixed formats (JSON, TXT, legacy backups). Validate schemas and isolate corrupted records. | Python OOP, JSON, Regex, Pathlib | [`Sprint1_Solution.ipynb`](Sprint1_Log_Analyzer/Sprint1_Solution.ipynb)<br/>`analytics_ready.json` (65 rows)<br/>`quarantine/` (7 files) |
| **[Sprint 2](Sprint2_Ecommerce_Dataset/)** | **E-Commerce Scraper & EDA** | Scrape live multi-page catalog with rate limiting. Clean dirty price strings, handle missing data, test price differentials ($p < 0.05$). | BeautifulSoup4, Requests, Pandas, Scipy, Seaborn | [`Sprint2_Solution.ipynb`](Sprint2_Ecommerce_Dataset/Sprint2_Solution.ipynb)<br/>`ecommerce_cleaned.csv` (103 rows)<br/>`eda_summary.png` (3-panel) |
| **[Sprint 3](Sprint3_SQL_Analysis/)** | **SQL Relational Engine** | Design 3NF relational schema. Deduplicate 595 reviews via `ROW_NUMBER() OVER`, rank products by category, and parse ISO dates. | SQLite3, SQLAlchemy, SQL Window Functions, CTEs | [`Sprint3_Solution.ipynb`](Sprint3_SQL_Analysis/Sprint3_Solution.ipynb)<br/>`ecommerce.db`<br/>Clean `reviews` (574 rows) |
| **[Sprint 4](Sprint4_PowerBI_Dashboard/)** | **Executive BI Dashboard** | Build business KPI canvas with DAX formulas (weighted average rating, discount impact). Deploy native Power BI desktop solution. | Power BI Desktop, DAX, Plotly, HTML5 Canvas | [`Sprint4_Ecommerce_Dashboard.pbix`](Sprint4_PowerBI_Dashboard/Sprint4_Ecommerce_Dashboard.pbix)<br/>[`Sprint4_Solution.ipynb`](Sprint4_PowerBI_Dashboard/Sprint4_Solution.ipynb)<br/>`DAX_Measures_Reference.md` |

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## Centralized Datasets Hub

All raw input data and intermediate benchmark files are centralized in [`datasets/`](datasets/README.md). Students and instructors can reset any sprint without touching the solutions.

```
datasets/
|-- README.md                              # Data dictionary & quick copy commands
|-- Sprint_1_Log_Analyzer/
|   `-- daily_logs/                        # 74 raw server logs (JSON, TXT, invalid)
|-- Sprint_2_Ecommerce_Dataset/
|   `-- ecommerce_raw_scraped.csv          # 149 raw scraped listings with currency signs
|-- Sprint_3_SQL_Analysis/
|   |-- ecommerce_cleaned.csv              # 103 cleaned products from Sprint 2
|   `-- reviews_raw.csv                    # 595 raw customer reviews (contains duplicates)
`-- Sprint_4_PowerBI_Dashboard/
    |-- products_for_powerbi.csv           # Denormalized product master
    |-- review_summary_for_powerbi.csv     # Product review aggregates
    `-- ecommerce.db                       # Direct SQLite reporting database
```

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## Repository File Structure

```
Data_Science_Notes/
|-- Complete Data Science - Project Based Learning.md  # Complete project specification
|-- README.md                                          # Master repository documentation
|-- .gitignore                                         # Git ignore specifications
|
|-- datasets/                                          # Central benchmark data hub
|   |-- README.md                                      # Data dictionary & setup guide
|   |-- Sprint_1_Log_Analyzer/                         # Raw server logs
|   |-- Sprint_2_Ecommerce_Dataset/                    # Raw scraped listings
|   |-- Sprint_3_SQL_Analysis/                         # Clean products & raw reviews
|   `-- Sprint_4_PowerBI_Dashboard/                    # BI reporting tables & database
|
|-- Sprint1_Log_Analyzer/                              # SPRINT 1: OOP File System Engine
|   |-- README.md                                      # Data placement & execution guide
|   |-- Sprint1_Solution.ipynb                         # Step-by-step Q&A solution notebook
|   |-- analytics_ready.json                           # Output (65 valid parsed records)
|   `-- data/                                          # Working data directory
|
|-- Sprint2_Ecommerce_Dataset/                         # SPRINT 2: Web Scraping & EDA
|   |-- README.md                                      # Data placement & execution guide
|   |-- Sprint2_Solution.ipynb                         # Step-by-step Q&A solution notebook
|   |-- eda_summary.png                                # 3-panel statistical EDA export
|   `-- data/                                          # Working data directory
|
|-- Sprint3_SQL_Analysis/                              # SPRINT 3: SQL Relational Database
|   |-- README.md                                      # Data placement & execution guide
|   |-- Sprint3_Solution.ipynb                         # Step-by-step Q&A solution notebook
|   |-- ecommerce.db                                   # SQLite normalized database
|   `-- data/                                          # Working data directory
|
`-- Sprint4_PowerBI_Dashboard/                         # SPRINT 4: Power BI & Executive Dashboard
    |-- README.md                                      # Data placement & execution guide
    |-- Sprint4_Ecommerce_Dashboard.pbix               # Native Power BI Desktop file
    |-- Sprint4_Solution.ipynb                         # Interactive Plotly notebook prototype
    |-- DAX_Measures_Reference.md                      # Complete DAX formula reference
    |-- dashboard_design_preview.html                  # Standalone interactive browser preview
    `-- data/                                          # Working data directory
```

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## Quick Start & Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/JayanGupta/Data_Science_Notes.git
cd Data_Science_Notes
```

### 2. Install Required Python Dependencies
```bash
pip install pandas numpy scipy matplotlib seaborn requests beautifulsoup4 sqlalchemy plotly nbconvert
```

### 3. Launch Jupyter Lab / Notebook
```bash
jupyter lab
```

Navigate to any sprint folder (e.g., `Sprint1_Log_Analyzer/Sprint1_Solution.ipynb`) and execute sequentially.

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## Contributing

Contributions, enhancements, and suggestions are welcome:

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/NewSprintFeature`)
3. Commit your Changes (`git commit -m "feat: enhance sprint analysis"`)
4. Push to the Branch (`git push origin feature/NewSprintFeature`)
5. Open a Pull Request

---

<img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif" width="100%">

## Show Your Support

Give a star if this repository helped your learning journey or classroom instruction.

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=JayanGupta/Data_Science_Notes&type=Date&theme=dark)](https://star-history.com/#JayanGupta/Data_Science_Notes&Date)

</div>

---

<div align="center">

<!-- Animated Footer Wave -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0d1117,50:1a1b27,100:00d9ff&height=120&section=footer" width="100%"/>

<br/>

<b>Engineered for Data Science Education by <a href="https://github.com/JayanGupta">Jayan Gupta</a></b>

<br/><br/>

<img src="https://forthebadge.com/images/badges/built-with-love.svg" />
<img src="https://forthebadge.com/images/badges/made-with-python.svg" />
<img src="https://forthebadge.com/images/badges/open-source.svg" />

</div>

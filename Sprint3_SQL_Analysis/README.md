# Sprint 3: Analyzing the E-Commerce Dataset in SQL

## Executive Overview

This project establishes the relational database tier of the curriculum. Learners ingest the cleaned e-commerce product dataset from Sprint 2 alongside raw customer review records, design a normalized 3-table database schema (`categories`, `products`, `reviews`), and execute analytical and cleaning workflows using advanced SQL: **Window Functions** (`RANK() OVER`), **Common Table Expressions (CTEs)**, **Safe Deduplication** (`ROW_NUMBER() OVER`), and **Date Standardization** to ISO `YYYY-MM-DD`.

The resulting SQLite database (`ecommerce.db`) connects directly to **Sprint 4 (Power BI Dashboarding)**.

---

## Data Architecture & File Placement

### 1. Data Sources & Download Locations
To set up or restore the inputs for this sprint, copy the datasets from the centralized repository hub:
- **Product Data:**
  - Source Path: `datasets/Sprint_3_SQL_Analysis/ecommerce_cleaned.csv` (or exported from Sprint 2)
  - Target Path: `Sprint3_SQL_Analysis/data/ecommerce_cleaned.csv`
- **Reviews Data:**
  - Source Path: `datasets/Sprint_3_SQL_Analysis/reviews_raw.csv`
  - Target Path: `Sprint3_SQL_Analysis/data/reviews_raw.csv`

### 2. Dataset Composition
- **`ecommerce_cleaned.csv` (103 Rows):** Clean, deduplicated product records containing `listing_id`, `name`, `brand`, `category`, `price`, `rating`, `review_count`.
- **`reviews_raw.csv` (595 Rows):** Customer review submissions containing intentional data hygiene problems:
  - **Duplicate Submissions:** ~4% identical review records (same product, date, rating, and text).
  - **Inconsistent Dates:** Dates formatted across 4 distinct styles (`YYYY-MM-DD`, `MM/DD/YYYY`, `YYYY.MM.DD`, `DD Month YYYY`).

---

## Directory Structure

### Expected Structure Before Execution
```text
Sprint3_SQL_Analysis/
├── README.md                          # Project documentation and operational guide
├── Sprint3_Solution.ipynb             # Executed solution notebook in Q&A format
└── data/
    ├── ecommerce_cleaned.csv          # Clean product listings from Sprint 2
    └── reviews_raw.csv                # 595 raw customer reviews
```

### Expected Structure After Execution
```text
Sprint3_SQL_Analysis/
├── README.md
├── Sprint3_Solution.ipynb
├── ecommerce.db                       # [OUTPUT] Normalized SQLite database
└── data/
    ├── ecommerce_cleaned.csv          # Clean product listings (unmodified)
    └── reviews_raw.csv                # Raw reviews (unmodified)
```

---

## How to Run the Project

### Option A: Interactive In-Class Coding (Jupyter Notebook)
Launch the solution notebook:
```bash
cd Sprint3_SQL_Analysis
jupyter notebook Sprint3_Solution.ipynb
```
- **Step 1 (Schema & Ingestion):** Creating normalized tables in SQLite via SQLAlchemy.
- **Step 2 (Window Functions):** Ranking products within categories via `RANK() OVER (PARTITION BY ... ORDER BY ...)`.
- **Step 3 (CTEs):** Building a multi-step Top Price Movers report with `WITH ... AS`.
- **Step 4 (In-Database Cleaning):** Deleting duplicate reviews using `ROW_NUMBER() OVER (...)`.
- **Step 5 (Date Standardization):** Converting messy date strings to ISO `YYYY-MM-DD`.

### Option B: Headless Batch Execution
Run the entire notebook non-interactively:
```bash
cd Sprint3_SQL_Analysis
jupyter nbconvert --to notebook --execute --inplace Sprint3_Solution.ipynb
```

---

## Verification & Output Audit

Following execution, verify database integrity:
1. **Database File:** `ecommerce.db` must exist and be accessible via SQLite.
2. **Table Counts:**
   - `categories`: exactly **3** records.
   - `products`: exactly **103** records.
   - `reviews`: reduced from **595** down to **574** deduplicated records (21 duplicates safely removed).
3. **Date Standardization:** All dates in `reviews.review_date` must match the ISO format `YYYY-MM-DD`.

### Quick Verification Command (PowerShell)
```powershell
python -c "import sqlite3; conn = sqlite3.connect('ecommerce.db'); c = conn.cursor(); assert c.execute('SELECT COUNT(*) FROM categories').fetchone()[0] == 3; assert c.execute('SELECT COUNT(*) FROM products').fetchone()[0] == 103; assert c.execute('SELECT COUNT(*) FROM reviews').fetchone()[0] == 574; print('[PASS] All Sprint 3 verification assertions passed!')"
```

---

## Environment Reset for Live Demonstrations

To reset the database before class so students can see it built live:
```powershell
# Delete the existing database file
Remove-Item -Path "ecommerce.db" -Force -ErrorAction SilentlyContinue

# Verify input CSVs are intact
Test-Path "data/ecommerce_cleaned.csv"
Test-Path "data/reviews_raw.csv"
```

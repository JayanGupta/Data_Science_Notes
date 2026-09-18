# Sprint 2: Building, Cleaning and Analyzing an E-Commerce Dataset

## Executive Overview

This project transitions learners from basic Python programming into the core scientific data stack (`NumPy`, `Pandas`, `SciPy`, `Matplotlib`, `Seaborn`). The project demonstrates how raw, noisy data scraped from e-commerce websites is ingested, sanitized through vectorized string cleaning and robust outlier detection, evaluated using inferential hypothesis testing, and visualized for executive review.

The resulting clean dataset forms the common foundation for **Sprint 3 (SQL Relational Storage)** and **Sprint 4 (Power BI Dashboarding)**.

---

## Data Architecture & File Placement

### 1. Data Source & Download Location
To set up or restore the initial environment, copy the raw scraped dataset from the centralized repository hub:
- **Source Path:** `datasets/Sprint_2_Ecommerce_Dataset/ecommerce_raw_scraped.csv`
- **Target Working Path:** `Sprint2_Ecommerce_Dataset/data/ecommerce_raw_scraped.csv`

### 2. Dataset Composition (149 Raw Listings)
The dataset represents real-world multi-category scraped listings across 3 departments (`electronics`, `home-goods`, `sports-outdoors`) containing common web scraping defects:
- **Price Inconsistencies:** Currency signs (`$`, `USD`), comma delimiters, and trailing whitespaces.
- **Missing Values:** ~6 unlisted prices, ~2 missing star ratings.
- **Scraper Duplicates:** ~5% duplicate product records caused by pagination re-visits.
- **Price Outliers:** Premium items in each category that require statistical detection via the Interquartile Range ($1.5 \times \text{IQR}$) method.
- **Brand Inconsistencies:** Arbitrary casing (`SONY`, `sony`, ` Sony `).

---

## Directory Structure

### Expected Structure Before Execution
```text
Sprint2_Ecommerce_Dataset/
├── README.md                          # Project documentation and operational guide
├── Sprint2_Solution.ipynb             # Executed solution notebook in Q&A format
└── data/
    └── ecommerce_raw_scraped.csv      # 149 raw scraped product listings
```

### Expected Structure After Execution
```text
Sprint2_Ecommerce_Dataset/
├── README.md
├── Sprint2_Solution.ipynb
├── eda_summary.png                    # [OUTPUT] 3-panel visualization figure
└── data/
    ├── ecommerce_raw_scraped.csv      # Raw input listings (unmodified)
    └── ecommerce_cleaned.csv          # [OUTPUT] 103 deduplicated, clean product records
```

---

## How to Run the Project

### Option A: Interactive In-Class Coding (Jupyter Notebook)
Launch the solution notebook:
```bash
cd Sprint2_Ecommerce_Dataset
jupyter notebook Sprint2_Solution.ipynb
```
- **Step 1 (Web Scraping):** Live scraping from the open sandbox `http://books.toscrape.com/`.
- **Step 2 (Data Cleaning):** Vectorized regex price cleaning and IQR outlier detection.
- **Step 3 (Hypothesis Testing):** Two-sample independent Welch's t-test comparing Electronics vs. Sports-Outdoors pricing.
- **Step 4 (Export):** Generating `data/ecommerce_cleaned.csv`.

### Option B: Headless Batch Execution
Run the entire notebook non-interactively:
```bash
cd Sprint2_Ecommerce_Dataset
jupyter nbconvert --to notebook --execute --inplace Sprint2_Solution.ipynb
```

---

## Verification & Output Audit

Following execution, verify the following artifacts:
1. **Cleaned Dataset:** `data/ecommerce_cleaned.csv` must exist and contain exactly **103** rows and 7 standardized columns (`listing_id`, `name`, `brand`, `category`, `price`, `rating`, `review_count`).
2. **Visualization Export:** `eda_summary.png` must be saved in the directory.
3. **Hypothesis Test:** The two-sample t-test must yield $p < 0.05$, demonstrating a statistically significant price difference between Electronics and Sports-Outdoors.

### Quick Verification Command (PowerShell)
```powershell
python -c "import pandas as pd, os; df = pd.read_csv('data/ecommerce_cleaned.csv'); assert len(df) == 103; assert os.path.exists('eda_summary.png'); print('[PASS] All Sprint 2 verification assertions passed!')"
```

---

## Environment Reset for Live Demonstrations

To reset the workspace for a clean live coding session:
```powershell
# Remove generated artifacts
Remove-Item -Path "data/ecommerce_cleaned.csv" -Force -ErrorAction SilentlyContinue
Remove-Item -Path "eda_summary.png" -Force -ErrorAction SilentlyContinue

# Verify raw CSV is intact
Test-Path "data/ecommerce_raw_scraped.csv"
```

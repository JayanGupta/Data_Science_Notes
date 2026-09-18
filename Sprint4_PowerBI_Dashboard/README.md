# Sprint 4: Dashboarding the E-Commerce Dataset in Power BI & BI Tools

## Executive Overview

This project completes the end-to-end data pipeline by delivering an executive business intelligence solution. Guided by the **Philosophy of Dashboarding**, learners define the Category Manager audience persona and 3 core business questions prior to visual design.

Learners explore:
1. Executive KPI cards (Total Products, Catalog Average Price, Review-Weighted Average Rating).
2. Comparative visual analytics (Category Price Spread, Price vs. Rating Satisfaction Matrix).
3. Enterprise DAX measures (`CALCULATE`, `ALLEXCEPT`, `AVERAGEX`, dynamic slicers).
4. Native Microsoft Power BI Desktop presentation via `Sprint4_Ecommerce_Dashboard.pbix`.

---

## Data Architecture & File Placement

### 1. Data Sources & Download Locations
To set up or restore the inputs for this sprint, copy the datasets from the centralized repository hub:
- **Products Catalog:**
  - Source Path: `datasets/Sprint_4_PowerBI_Dashboard/products_for_powerbi.csv`
  - Target Path: `Sprint4_PowerBI_Dashboard/data/products_for_powerbi.csv`
- **Reviews Summary:**
  - Source Path: `datasets/Sprint_4_PowerBI_Dashboard/review_summary_for_powerbi.csv`
  - Target Path: `Sprint4_PowerBI_Dashboard/data/review_summary_for_powerbi.csv`
- **SQLite Database (Alternative Direct Connection):**
  - Source Path: `datasets/Sprint_4_PowerBI_Dashboard/ecommerce.db`
  - Target Path: `Sprint4_PowerBI_Dashboard/data/ecommerce.db`

### 2. Dataset Composition
- **`products_for_powerbi.csv` (103 Rows):** Denormalized product listings with inline category names for zero-overhead Power BI CSV import.
- **`review_summary_for_powerbi.csv` (103 Rows):** Product review aggregates (`product_id`, `actual_review_count`, `avg_review_rating`, `first_review_date`, `latest_review_date`) designed for relational modeling (`1:1` relationship on `product_id`).
- **`ecommerce.db`:** Full relational database from Sprint 3 for instructors demonstrating direct database connectivity via ODBC.

---

## Directory Structure

```text
Sprint4_PowerBI_Dashboard/
├── README.md                          # Project documentation and operational guide
├── Sprint4_Ecommerce_Dashboard.pbix   # [PRIMARY SOLUTION] Native Power BI Desktop solution file
├── Sprint4_Solution.ipynb             # [PYTHON PROTOTYPE] Executed solution notebook in Q&A format
├── DAX_Measures_Reference.md          # Copy-paste reference of pre-written DAX formulas
├── dashboard_design_preview.html      # Zero-install standalone interactive browser preview
└── data/
    ├── products_for_powerbi.csv       # 103 denormalized product records
    ├── review_summary_for_powerbi.csv # 103 review aggregation records
    └── ecommerce.db                   # SQLite database source
```

---

## How to Run and Present the Project

### Option 1: Microsoft Power BI Desktop (Native Solution)
Double-click `Sprint4_Ecommerce_Dashboard.pbix` to launch in Power BI Desktop:
1. **Model View:** Inspect the `1:1` relationship between `products` and `review_summary` on `product_id`.
2. **Data View:** Review pre-built DAX measures (`Average Price`, `Total Products`, `Above-Category-Average Flag`, `Weighted Avg Rating`).
3. **Report View:** Demonstrate the 3 KPI cards, category comparison bar chart, price-vs-rating scatter chart, and interactive price slicer.

### Option 2: Interactive Python Solution (Jupyter Notebook)
Run the Python solution notebook to demonstrate prototype dashboarding with Plotly:
```bash
cd Sprint4_PowerBI_Dashboard
jupyter notebook Sprint4_Solution.ipynb
```

### Option 3: Zero-Install Standalone Browser Preview
Open `dashboard_design_preview.html` in Chrome, Edge, or Firefox. This provides an interactive HTML/CSS/JS mockup of the target dashboard with functional price filtering.

---

## Verification & Output Audit

To verify that all Sprint 4 components are functioning properly:
1. **Power BI Solution:** `Sprint4_Ecommerce_Dashboard.pbix` exists and opens cleanly in Power BI Desktop (`PBIDesktop.exe`).
2. **Notebook Execution:** `Sprint4_Solution.ipynb` executes cleanly with 0 errors and displays all Plotly charts.
3. **Data Parity:** Total active products must equal exactly **103**, with an average price of **$116.89**.

### Quick Verification Command (PowerShell)
```powershell
python -c "import pandas as pd; df = pd.read_csv('data/products_for_powerbi.csv'); assert len(df) == 103; assert round(df['price'].mean(), 2) == 116.89; print('[PASS] All Sprint 4 verification assertions passed!')"
```

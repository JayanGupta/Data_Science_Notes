# Centralized Datasets Hub (Sprints 1–4)

Welcome to the centralized datasets repository. This folder contains all the raw, benchmark, and cleaned datasets used throughout **Sprints 1 to 4**. 

If you are a student or instructor setting up the projects from scratch, you can copy the required files from here directly into each sprint's local `data/` folder.

---

## Datasets Directory Map

```text
datasets/
├── Sprint_1_Log_Analyzer/
│   └── daily_logs/                           # 75 simulated server log files (.txt and .json)
│       ├── event_001.json ... event_040.json # Valid JSON logs
│       ├── event_041.txt ... event_065.txt   # Valid CSV-formatted text logs
│       ├── event_066.json ... event_072.txt  # 7 intentionally corrupted files (corrupt syntax, missing fields)
│       └── .DS_Store, ~backup..., notes.md   # 3 hidden/junk files for filtering practice
│
├── Sprint_2_Ecommerce_Dataset/
│   └── ecommerce_raw_scraped.csv             # 149 raw scraped product listings across 3 categories
│
├── Sprint_3_SQL_Analysis/
│   ├── ecommerce_cleaned.csv                 # 103 cleaned product listings from Sprint 2
│   └── reviews_raw.csv                       # 595 customer review records with duplicates and messy dates
│
└── Sprint_4_PowerBI_Dashboard/
    ├── products_for_powerbi.csv              # Denormalized products table for Power BI import
    ├── review_summary_for_powerbi.csv        # Product review aggregates table (join key: product_id)
    └── ecommerce.db                          # Clean SQLite database from Sprint 3 for direct DB connection
```

---

## Data Dictionaries

### 1. `Sprint_1_Log_Analyzer/daily_logs/`
Simulates distributed microservice server traffic.
- **JSON Format**: `{"ip_address": "192.168.1.10", "status_code": 200, "timestamp": "..."}`
- **TXT Format**: `192.168.1.15,500` (comma-separated: `ip,status_code`)
- **Intentional Errors**:
  - `JSONDecodeError`: Corrupted JSON syntax.
  - `KeyError`: Missing `ip_address` or `status_code` key.
  - `ValueError`: Missing comma in `.txt` files.

### 2. `Sprint_2_Ecommerce_Dataset/ecommerce_raw_scraped.csv`
Simulates raw web scraping output from an e-commerce catalog.
| Column | Type | Description | Messiness / Issues |
|---|---|---|---|
| `listing_id` | Integer | Scraped listing ID | Sequential identifier |
| `name` | String | Product title | Whitespace variations |
| `brand` | String | Manufacturer brand | Inconsistent casing (`SONY`, `sony`, ` Sony `) |
| `category` | String | Catalog category | `electronics`, `home-goods`, `sports-outdoors` |
| `price` | String | Listing price | Mixed formatting (`$199.99`, `USD 199.99`, blanks) |
| `rating` | Float/String | Star rating (1–5) | Missing values (`NaN`) |
| `review_count` | Integer | Total customer reviews | Range: 0 to 5,000+ |

### 3. `Sprint_3_SQL_Analysis/reviews_raw.csv`
Simulates real customer review submissions.
| Column | Type | Description | Messiness / Issues |
|---|---|---|---|
| `review_id` | Integer | Unique review identifier | Primary key candidate |
| `listing_id` | Integer | Associated product ID | Foreign key to `products` table |
| `review_date` | String | Submission date | **4 formats**: `YYYY-MM-DD`, `MM/DD/YYYY`, `YYYY.MM.DD`, `DD Month YYYY` |
| `rating` | Integer | Star rating awarded | 1 to 5 stars |
| `review_text` | String | Written user review | Contains ~4% exact duplicate submissions |

### 4. `Sprint_4_PowerBI_Dashboard/`
Denormalized reporting exports generated from the Sprint 3 SQLite database.
- `products_for_powerbi.csv`: Flat product catalog with inline category names.
- `review_summary_for_powerbi.csv`: Aggregated reviews summary containing `actual_review_count`, `avg_review_rating`, `first_review_date`, and `latest_review_date`.
- `ecommerce.db`: SQLite database with `categories`, `products`, and `reviews` tables.

---

## Quick Copy Commands (PowerShell)

If you want to reset any sprint's data folder using this central hub:

```powershell
# Reset Sprint 1
Copy-Item -Path "datasets/Sprint_1_Log_Analyzer/daily_logs" -Destination "Sprint1_Log_Analyzer/data/daily_logs" -Recurse -Force

# Reset Sprint 2
Copy-Item -Path "datasets/Sprint_2_Ecommerce_Dataset/ecommerce_raw_scraped.csv" -Destination "Sprint2_Ecommerce_Dataset/data/ecommerce_raw_scraped.csv" -Force

# Reset Sprint 3
Copy-Item -Path "datasets/Sprint_3_SQL_Analysis/*" -Destination "Sprint3_SQL_Analysis/data/" -Recurse -Force

# Reset Sprint 4
Copy-Item -Path "datasets/Sprint_4_PowerBI_Dashboard/*" -Destination "Sprint4_PowerBI_Dashboard/data/" -Recurse -Force
```

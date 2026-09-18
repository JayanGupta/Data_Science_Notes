# **Sprint 1: Python and File Management Fundamentals**  

* Python Fundamentals   
  * Getting Started with Python  
  * Conditionals and Flow Control  
  * Data Structures  
  * Function & OOP  
  * Special Functions  
  * Exceptions  
  * Pythonic Style Guide  
* File Management System  
  * Getting Started with Files  
  * Recorded Project: Inventory Management System with Files  
  * OS with Shell

**Live Project Class: Automated Data Pipeline & Forensic Log Analyzer**

Learners will step into the shoes of a Data Analyst. They will be provided with a folder containing dozens of messy, simulated daily server logs (or IoT sensor data) in a mix of TXT and JSON formats, some of which are intentionally corrupted.

* Write a script that scans a specific local directory, identifies the data files, and ignores hidden or irrelevant junk files.  
* Build a DataParser class that is responsible for opening and inspecting the contents of each file.  
* Use try-except blocks to gracefully catch malformed JSONs or missing data fields without crashing the entire script.  
* Use dictionaries and lists to aggregate the data and count anomalies (e.g., "Which IP address threw the most errors?" or "Which sensor had missing values?").  
* Consolidate all the valid, cleaned data into a single analytics\_ready.json file, and use the os module to automatically move any corrupted files into a newly created quarantine folder.

# **Sprint 2: Statistics, Probability, and Data Analysis with Python**

* Statistics & Probability  
  * Descriptive Statistics  
  * Sampling  
  * Hypothesis Testing  
  * Correlation and Covariance  
  * Probability Distributions  
  * Bayes Theorem  
* Data Collection via Web Scraping  
  * Introduction to Web Scraping  
  * Recorded Project: Book Scraper  
  * Recorded Project: Wikipedia Scraper  
  * Recorded Project: Youtube Scrapper  
  * Recorded Project: Image Dataset Creation  
  * Common Selenium Setup Issues  
* Data Analysis with NumPy  
  * NumPy Arrays  
  * Essential Array Operations  
  * Working With Files  
  * Data Cleaning and Analysis with NumPy  
  * Visualization with NumPy  
* Data Analysis with Pandas  
  * Principles of Exploratory Data Analysis  
  * Data Analysis with Pandas  
* Data Visualization with Python  
  * Data Visualization with Matplotlib  
  * Data Visualization with Seaborn  
* Data Analysis Projects  
  * Recorded Project: Article Content Analysis  
  * Recorded Project: IMDB Movies Data Analysis  
  * Recorded Project: The Bookwarm's Data Guide  
  * Recorded Project: Data Storytelling: Analysing Survival on the Titanic  
  * Recorded Project: Cracking the Code: An Inside Look at Netflix's Content Strategy  
* Other Python Libraries for Data Analysis  
  * Pandas AI  
  * YData Profiling  
  * Streamlit in One Video

**Live Project Class: Building, Cleaning and Analyzing an E-Commerce Dataset**

* A scraper (BeautifulSoup/Selenium) pulling product listings from a chosen e-commerce category — name, price, rating, review count, category  
* NumPy-based cleaning: handling nulls, outlier detection, fixed-size array operations  
* Pandas EDA: univariate analysis, duplicate handling, groupby summaries by category/brand  
* Descriptive statistics \+ hypothesis testing: is there a significant price difference between two brands/categories? (t-test, using the Sampling/CLT/P-values content)  
* Correlation analysis: does rating correlate with price or review count?  
* Matplotlib/Seaborn visualizations: distribution plots, box plots for outliers, correlation heatmap

# 

# **Sprint 3: Data Analysis with Databases and Excel**

* Data Analysis with SQL  
  * SQL-Fundamentals  
  * Data Retrieval with SQL  
  * Advanced SQL Techniques  
  * Data Cleaning with SQL  
  * Query Optimisation Techniques  
* Data Analysis with NoSQL  
  * MongoDB Concepts: NoSQL Paradigm  
* Python Integration with Databases  
  * Python Integration with Databases  
* Data Analysis with Excel  
  * Exploring Data  
  * Preparing Data  
  * Analysing Data  
* Excel Projects  
  * Recorded Project: Building a Netflix Analytics Dashboard Using Excel  
  * Recorded Project: Healthcare Intelligence Dashboard  
  * Recorded Project: End-to-End Retail Business Analytics Using Excel

**Live Project Class: Analyzing the E-Commerce Dataset in SQL**

* Design a normalized schema (products, categories, reviews tables) and load the Sprint 2 dataset into SQLite via SQLAlchemy  
* Write analytical SQL: window functions for ranking products within category, CTEs for a "top movers" report, joins across tables  
* SQL-based data cleaning: dedupe, outlier handling, date standardization

# **Sprint 4: Data Dashboarding and Business Intelligence**

* Principles of Dashboarding  
  * Principles of Dashboarding  
* Dashboarding with Plotly Dash  
  * Plotly Dash  
* Tableau  
  * Introduction to Tableau  
  * Data in Tableau  
  * The Working Area in Tableau  
  * Customizing Your Data  
  * Working on Charts  
  * Creating Your First Dashboard in Tableau  
* Power BI  
  * Introduction to Power BI  
  * Working with Data in Power BI  
  * Visuals in Power BI  
  * Creating Your First Dashboard in Power BI

**Live Project Class: Dashboarding the E-Commerce Dataset in Power BI**

* Apply "Philosophy of Dashboarding" principles first: define the audience (category manager) and the 3 questions the dashboard must answer before building anything  
* KPI cards (avg price, avg rating, total products)  
* Category/brand comparison charts  
* Filters/slicers by category and price range  
* Calculated measures with DAX; Tableau track: calculated fields \+ parameters  
* Publish/export the finished dashboard

# **Sprint 5: Supervised Machine Learning**

* Maths for Data Science  
  * Vectors and Matrices  
  * Dot Product  
  * Linear Transformations  
  * Eigenvectors and Eigenvalues  
  * Matrix Decomposition  
  * Differentiation  
  * Application of Maths in Data Science  
  * Maths of k-means  
  * Maths of Support Vector Machines  
  * Maths of Decision Trees  
  * Maths of Artificial Neural Networks  
* Machine Learning  
  * Introduction to Machine Learning  
  * Linear Regression  
  * Bias-Variance Tradeoff and Regularization  
  * Logistic Regression  
  * Decision Trees  
  * Naive Bayes  
  * Support Vector Machines  
  * K Nearest Neighour

**Live Project Class: Customer Churn & Revenue Prediction**

* Build a regression model predicting customer lifetime value using Linear Regression  
* Using Evaluation Metrics: comparing MSE vs MAE  
* Build a  classification model predicting churn using Logistic Regression, Decision Tree, KNN, and Naive Bayes  
* Build a  model comparison table: accuracy, confusion matrix, precision/recall tradeoffs

# **Sprint 6: Unsupervised & Ensemble Learning  and Domain Specializations**

* Unsupervised Learning  
  * Clustering  
  * Dimensionality Reduction  
* Ensemble Learning  
  * AdaBoost (Adaptive Boosting)  
  * Gradient Boosting  
  * XGBoost (Extreme Gradient Boosting)  
* Other ML Techniques  
  * Anomaly Detection  
  * Time Series Forecasting  
  * Challenges of Machine Learning  
  * Mastering ML Interviews  
* Domain Specializations  
  * An Introduction to Natural Language Processing  
  * Mastering Strings & ASCII  
  * Getting Started with Spacy  
  * Information Extraction Techniques  
  * Next Word Prediction \- Probabilistic Model  
  * Text Encoding Techniques  
  * Image Enhancement: An Introduction to Fundamentals of Image Processing  
  * Image Processing Techniques  
  * Capturing Selfies using OpenCV  
  * Image Manipulation  
  * Create Your Own Instagram Filters  
  * Image Masking Techniques  
  * Face Detection and Manipulation with OpenCV  
* ML Projects  
  * Recorded Project: Predicting Housing Market Trends with AI  
  * Recorded Project: AI in Healthcare: Building a Heart Disease Predictor  
  * Recorded Project: Preventing Customer Churn with Feature Engineering  
  * Recorded Project: Predicting Future with Time Series AI  
  * Recorded Project: Unlocking Customer Personas with AI

**Live Project Class: Customer Segmentation & Review Intelligence**

* K-Means clustering to segment customers/products (with Silhouette Score to validate cluster count), plus PCA to visualize high-dimensional segments in 2D  
* An XGBoost model predicting which segment will churn or generate the most revenue next quarter — reinforcing bagging vs boosting vs stacking conceptually first  
* A lightweight NLP add-on: tokenize and TF-IDF-encode product reviews, run a quick sentiment classifier (Naive Bayes on TF-IDF), and pull out top complaint/praise keywords per cluster

# **Sprint 7: Neural Network Fundamentals**

* Neural Networks  
  * Introduction and Historical Context  
  * Neural Network Architecture  
  * The Learning Process: Training Neural Networks  
  * Model Evaluation  
  * Addressing Training Challenges  
  * Improving Model Performance  
* Pytorch  
  * Deep Learning with Pytorch  
* CNNs and RNNs  
  * Convolutional Neural Networks  
  * Recurrent Neural Networks

**Live Project Class: Apple Grade Classifier**

* Utilize Kaggle's dataset of \~600 apple images of "Fruits fresh and rotten apples sorted into 3 classes: Grade A (unblemished), Grade B (minor bruising), Reject (spoiled)  
* Build A CNN with 3 conv blocks (32→64→128 filters) built from scratch in PyTorch — no pretrained weights  
* Train it twice: once on raw images, once with augmentation (±15° rotation, horizontal flip, 0.8–1.2x zoom) — plot both validation curves on the same graph  
* Evaluate the two models using recall. Understand why accuracy may not be a good metric.

# **Sprint 8: Deep Learning**

* Significant Deep Learning Advancements  
  * Transfer Learning  
  * Significant Computer Vision Models  
  * Significant Natural Language Processing Models  
  * Other Significant Models & Architectures  
* Deep Learning Projects  
  * Recorded Project: Building a Digit Recognizer from Scratch  
  * Recorded Project: Fast-Tracking Image Classification with Transfer Learning  
  * Recorded Project: One-Line Solutions with Hugging Face Pipelines  
  * Recorded Project: Building a Real-Time Object Detector  
  * Recorded Project: Applying Deep Learning to Time Series Data  
  * Recorded Project: Build Your Own GPT

**Live Project Class: Review Sentiment Analysis**

* Reuse the actual product-review data from the e-commerce dataset built back in Sprints 2–4  
* Model A: Bag-of-Words \+ RNNs model  
* Model B: reviews represented as averaged 100-dim GloVe vectors, fed into the same RNNs model  
* Head-to-head on the same train/test split: accuracy, F1

# **Sprint 9: Prompt Engineering**

* Foundations of Gen AI  
  * Gen AI Glossary for Beginners  
  * Transformers \- Conceptual Foundations  
  * Hugging Face Basics  
  * Benchmarks and Leaderboards  
* Prompt Engineering  
  * Prompt Engineering  
  * LangChain for Prompt Engineering  
* Prompt Engineering Projects  
  * Recorded Project: Data Analysis with AI  
  * Recorded Project: Building a Natural Language to SQL Generator  
  * Recorded Project: Building a Smart OCR Bot

**Live Project Class: Learner Support Ticket Router**

* A set of 40 learner support tickets across 4 categories: "video/access issue," "certificate not issued," "refund request," "content bug report" — each needing a category \+ urgency (High/Medium/Low) label  
* A LangChain PromptTemplate with zero-shot learning \+ a Pydantic output parser so the result is always valid structured JSON, never free text  
* A LangChain PromptTemplate with few-shot prompting (one per category) \+ a Pydantic output parser so the result is always valid structured JSON, never free text  
* A LangChain PromptTemplate with unified prompt approach \+ a Pydantic output parser so the result is always valid structured JSON, never free text  
* Run the same 20 held-out tickets through the three prompts for comparison

# **Sprint 10: Retrieval Augmented Generation**

* Embeddings & Vector Databases  
  * Embeddings  
  *  Vector Databases  
* Retrieval Augmented Generation  
  * Theory of RAG  
  * LangChain for RAG  
  * Variations of RAG  
* RAG Projects  
  * Recorded Project: Build Your Own Intelligent Image Search Engine  
  * Recorded Project: Chat with Your Knowledge Base  
* Local Gen AI  
  * Local Model Management with Ollama  
  * Multi-Modal Applications in Gen AI

**Live Project Class: Chat With a PDF That Has Charts**

* Ingest a document that mixes text and charts/diagrams (a report or slide deck works well)  
* Split into two retrieval paths: text chunks embedded normally, images/diagrams embedded separately (CLIP-style) — both go into the same vector DB  
* Build a retriever that can return a relevant diagram alongside the text answer, not just text  
* Run the entire pipeline locally through Ollama — no external API calls, so it's demo-safe and cost-free  
* A/B test two RAG variants (e.g., baseline vs. hybrid search, or vs. re-ranking) and compare retrieval quality on the same question set

# **Sprint 11: Agentic AI**

* Basics of LLMOps  
  * ChainLit & Gradio  
  * LLM Guardrails  
  * LLM Evaluation Strategies  
  * LangSmith for Observability  
* Multi-Agent Systems & Frameworks  
  * Introduction to CrewAI  
  * Multi-Agent Collaboration Patterns  
  * Memory and Tool Calling in Agents  
  * Agent Reflection and Self-Correction  
  * Human-in-the-Loop Workflows  

**Live Project Class: Autonomous Multi-Agent Research Assistant**

* Build a multi-agent team using CrewAI / LangGraph consisting of a Senior Researcher agent and a Technical Writer agent  
* Configure web search and documentation tools for autonomous information retrieval  
* Orchestrate sequential task delegation and critique loops to synthesize comprehensive research reports  
* Add human-in-the-loop validation checkpoints before finalizing generated artifacts  

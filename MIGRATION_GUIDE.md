# Migration Guide: AML to NLP Project

## Overview
This document explains how the project was transformed from an AML (Anti-Money Laundering) transaction monitoring system to a LinkedIn Job Analysis NLP platform.

## 🔄 Conceptual Mapping

### Core Concepts
| AML Concept | NLP Equivalent | Purpose |
|-------------|----------------|---------|
| Transaction | Job Posting | Unit of analysis |
| Anomaly Detection | Entity Extraction | Finding patterns |
| Risk Score | Similarity Score | Ranking/matching |
| Alert Investigation | Resume Matching | Finding relevant items |
| Feature Engineering | Text Preprocessing | Data preparation |
| Model Ensemble | Multi-model NLP | Comprehensive analysis |

### Data Structure
| AML Field | NLP Field | Description |
|-----------|-----------|-------------|
| TRANSACTION_KEY | Job Link | Unique identifier |
| DATE_KEY | Date | Temporal information |
| CURRENCY_AMOUNT | (N/A) | Removed - not applicable |
| byorder_id | Company | Entity identifier |
| MECHANISM_DESC | Job Title | Transaction type |
| hbos_anomaly_score | NER confidence | Model score |
| pca_isolation_forest_score | Topic probability | Model score |
| Description | Description | Text field |

## 🏗️ Architecture Changes

### Before (AML)
```
Streamlit App
├── Home.py (Transaction dashboard)
├── pages/1_EDA.py (Transaction analysis)
├── pages/2_AML_Analytics.py (Anomaly detection)
├── functions/database.py (SQL queries)
├── functions/components.py (Risk cards)
└── aml/ (ML models)
    ├── dataset/
    ├── models/
    └── notebooks/
```

### After (NLP)
```
Streamlit App
├── Home.py (Job market dashboard)
├── pages/1_EDA.py (Job analysis)
├── pages/2_NLP_Analytics.py (NLP features)
├── functions/nlp_database.py (Data loading)
├── functions/nlp_components.py (Visualizations)
└── workspace/ (NLP analysis)
    ├── Data/
    ├── NER/
    ├── Topic Modeling/
    └── Word Embedding/
```

## 📦 Dependency Changes

### Removed (AML-specific)
```python
# Machine Learning for fraud detection
xgboost
catboost
lightgbm
vecstack

# Database connectors
psycopg2-binary
sqlalchemy

# Document processing
unstructured
unstructured-client
unstructured-inference
pdfminer
opencv-python
pikepdf

# Other
np_utils
graphviz
dash-ag-grid
pycountry
cartopy
```

### Added (NLP-specific)
```python
# NLP core libraries
spacy
nltk
gensim

# Transformers and embeddings
transformers
sentence-transformers
torch
torchvision
torchaudio

# Visualization
wordcloud

# Data collection
linkedin-jobs-scraper
```

## 🔧 Function Mapping

### Database Functions

#### AML → NLP Transformation

**AML: `functions/database.py`**
```python
def execute_query(query, params=None)
def get_transaction_counts()
def get_transactions_above_threshold(threshold, model)
def get_similar_transactions(transaction_id)
```

**NLP: `functions/nlp_database.py`**
```python
def load_job_data(workspace_path)
def get_job_by_id(df, job_id)
def search_jobs(df, query, search_columns)
def filter_by_company(df, company)
```

### Visualization Components

**AML: `functions/components.py`**
```python
def create_risk_cards(high_risk, medium_risk, low_risk)
def create_transaction_pattern_analysis(df)
def create_risk_time_series_plot(df)
def create_transaction_table(df, columns)
```

**NLP: `functions/nlp_components.py`**
```python
def create_keyword_analysis(df, keywords)
def create_company_distribution(df, top_n)
def create_job_title_distribution(df, top_n)
def display_job_metrics(df)
```

## 📊 Page Structure Changes

### Home Page

**Changes:**
- Title: "AML Analysis Platform" → "LinkedIn Job Intelligence Platform"
- Icon: 🏦 → 💼
- Metrics: Transactions/Alerts → Jobs/Companies
- Navigation: Anomaly Detection → NLP Analytics

### EDA Page

**Before (AML):**
- Load AML dataset from database
- Transaction distributions
- Risk score analysis
- Model performance metrics

**After (NLP):**
- Load job data from CSV/JSON
- Job market statistics
- Company and title distributions
- Description length analysis

### Analytics Page

**Before (AML):**
- Anomaly detection models (HBOS, PCA+IF)
- Risk threshold calibration
- Transaction investigation
- Alert generation

**After (NLP):**
- Named Entity Recognition
- Topic Modeling (LDA/LSA)
- Word Embeddings (Word2Vec/SBERT)
- Resume Matching

## 🎯 Feature Comparison

### AML Features → NLP Features

| AML Feature | NLP Equivalent | Method |
|-------------|----------------|--------|
| Anomaly Detection | Entity Extraction | spaCy NER |
| Risk Scoring | Similarity Scoring | Cosine similarity |
| Transaction Clustering | Topic Discovery | LDA/LSA |
| Alert Investigation | Resume Matching | SBERT embeddings |
| Time Series Analysis | Trend Analysis | Temporal patterns |
| Network Analysis | (Future) Company Networks | Graph analysis |

## 💾 Data Pipeline Changes

### AML Pipeline
```
Database (PostgreSQL)
  ↓
SQL Queries
  ↓
Pandas DataFrame
  ↓
Feature Engineering
  ↓
ML Models (HBOS, PCA+IF)
  ↓
Risk Scores
  ↓
Streamlit Dashboard
```

### NLP Pipeline
```
LinkedIn Scraper
  ↓
CSV Files (scraps/)
  ↓
Data Cleaning & Combining
  ↓
JSON/CSV (workspace/Data/)
  ↓
NLP Processing (NER, Topics, Embeddings)
  ↓
Analysis Results
  ↓
Streamlit Dashboard
```

## 🔍 Code Patterns

### Loading Data

**AML Pattern:**
```python
from functions.database import execute_query, SELECTED_COLUMNS

df = execute_query("SELECT * FROM transactions WHERE risk_score > ?", [threshold])
```

**NLP Pattern:**
```python
from functions.nlp_database import load_job_data

df = load_job_data(workspace_path)
```

### Creating Visualizations

**AML Pattern:**
```python
from functions.components import create_risk_cards

create_risk_cards(high_risk_count, medium_risk_count, low_risk_count)
```

**NLP Pattern:**
```python
from functions.nlp_components import create_keyword_analysis

create_keyword_analysis(df, keywords=['python', 'java', 'sql'])
```

### Filtering Data

**AML Pattern:**
```python
# SQL-based filtering
query = "SELECT * FROM transactions WHERE amount > ? AND risk_score > ?"
df = execute_query(query, [10000, 0.8])
```

**NLP Pattern:**
```python
# Pandas-based filtering
from functions.nlp_database import filter_by_company, search_jobs

df = filter_by_company(df, "Google")
df = search_jobs(df, "machine learning", ['Job Title', 'Description'])
```

## 🧪 Testing Strategy

### AML Testing
- Test database connections
- Validate SQL queries
- Check model predictions
- Verify risk score calculations

### NLP Testing
- Test data loading from multiple sources
- Validate text preprocessing
- Check NER entity extraction
- Verify embedding computations
- Test resume parsing

## 📈 Performance Considerations

### AML Optimizations
- Database indexing
- Query optimization
- Model caching
- Batch processing

### NLP Optimizations
- Model loading (load once, cache)
- Embedding precomputation
- Text preprocessing caching
- Parallel processing for large datasets
- GPU acceleration for transformers

## 🚀 Deployment Differences

### AML Deployment
- Database server required
- Model files on server
- Secure connection to database
- Regular model retraining

### NLP Deployment
- No database required
- Model files packaged with app
- Local file system access
- Data updated via scraper

## 📚 Documentation Updates

### Updated Files
1. ✅ `README.md` - Complete rewrite for NLP project
2. ✅ `QUICKSTART.md` - New quick start guide
3. ✅ `UPDATE_SUMMARY.md` - Detailed change log
4. ✅ `MIGRATION_GUIDE.md` - This document

### Code Documentation
- All functions now have NLP-focused docstrings
- Examples updated to show job analysis
- Comments reference job postings instead of transactions

## 🎓 Skills Transfer

### Concepts That Transfer
1. **Data Loading**: SQL → CSV/JSON
2. **Visualization**: Plotly remains the same
3. **UI/UX**: Streamlit patterns unchanged
4. **State Management**: Session state usage identical
5. **Error Handling**: Try/except patterns similar

### New Skills Required
1. **NLP Fundamentals**: Tokenization, lemmatization
2. **spaCy**: Entity recognition, text processing
3. **Gensim**: Topic modeling with LDA/LSA
4. **Transformers**: BERT, Sentence-BERT
5. **Text Embeddings**: Word2Vec, document vectors

## 🔮 Future Enhancements

### Potential Additions
1. **Advanced NER**: Custom entity types for job-specific info
2. **Deep Learning**: Fine-tuned BERT for classification
3. **Knowledge Graphs**: Company-skill-job relationships
4. **Real-time Scraping**: Live job market monitoring
5. **Recommendation System**: Personalized job suggestions
6. **Salary Prediction**: ML model for salary estimation
7. **Skill Gap Analysis**: Compare resume vs requirements
8. **Career Path Mapping**: Transition recommendations

## ✅ Migration Checklist

- [x] Update Home page title and branding
- [x] Replace AML metrics with job metrics
- [x] Create new EDA page for job data
- [x] Build NLP Analytics page
- [x] Write nlp_database.py functions
- [x] Create nlp_components.py visualizations
- [x] Update requirements.txt for NLP
- [x] Update utils.py workspace path
- [x] Rewrite README.md
- [x] Create QUICKSTART.md
- [x] Document changes in UPDATE_SUMMARY.md
- [ ] Test data loading from all sources
- [ ] Implement actual NER model calls
- [ ] Integrate topic modeling code
- [ ] Add word embedding functionality
- [ ] Build resume matching feature
- [ ] Add data export functionality
- [ ] Create unit tests
- [ ] Deploy to production

## 🎯 Key Takeaways

1. **Conceptual Similarity**: Both projects analyze patterns in data (transactions vs jobs)
2. **Technical Adaptation**: Core libraries changed (SQL → NLP), but framework (Streamlit) remained
3. **Architecture Preservation**: Multi-page app structure maintained
4. **Data Flow**: Both follow ETL pattern (Extract → Transform → Load → Analyze)
5. **User Experience**: Dashboard-style interface with navigation and visualizations

## 📞 Support

For questions about the migration:
- Review `UPDATE_SUMMARY.md` for detailed changes
- Check `QUICKSTART.md` for usage instructions
- Refer to inline code comments
- Consult original Jupyter notebooks in `workspace/`

---

**Migration Status**: ✅ Core structure complete, ready for NLP model integration

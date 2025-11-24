# nlp-fall-2025

## LinkedIn Job Analysis Platform

This project is an NLP-powered platform for analyzing LinkedIn job postings using advanced natural language processing techniques including Named Entity Recognition (NER), Topic Modeling, and Word Embeddings.

## 🚀 Features

### 1. Job Data Collection
- **LinkedIn Job Scraper**: Automated scraping of job postings from LinkedIn
- **70+ Job Titles**: Comprehensive coverage across multiple industries
- **Structured Data**: Clean CSV format with job details, descriptions, and metadata

### 2. NLP Analysis Tools
- **Named Entity Recognition (NER)**: Extract skills, technologies, qualifications, and entities
- **Topic Modeling**: Discover themes using LDA and LSA
- **Word Embeddings**: Word2Vec and Sentence-BERT for semantic analysis
- **Resume Matching**: Match resumes to job descriptions

### 3. Interactive Dashboard
- **Streamlit Web App**: User-friendly interface for data exploration
- **EDA Visualizations**: Interactive charts and statistics
- **NLP Analytics**: Run NLP models and view results
- **Real-time Analysis**: Process job descriptions on-demand

## 📁 Project Structure

```
nlp-fall-2025/
├── app-streamlit/          # Streamlit web application
│   ├── Home.py            # Main dashboard
│   ├── pages/
│   │   ├── 1_EDA.py       # Exploratory data analysis
│   │   ├── 2_NLP_Analytics.py  # NLP analysis tools
│   │   └── 3_Meet_Our_Team.py
│   ├── functions/         # Helper functions
│   │   ├── nlp_database.py      # Data loading utilities
│   │   └── nlp_components.py    # Visualization components
│   └── requirements-nlp.txt     # Python dependencies
├── workspace/             # Analysis notebooks and data
│   ├── Data/             # Job datasets
│   ├── NER/              # Named Entity Recognition
│   ├── Topic Modeling/   # LDA and LSA implementations
│   ├── Word Embedding/   # Word2Vec and SBERT
│   └── Resume_testing/   # Resume matching experiments
├── scraps/               # Raw scraped data (CSV files)
└── linkedin.py           # LinkedIn scraper script
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Chrome browser (for scraping)
- LinkedIn account

### Setup Steps

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/nlp-fall-2025.git
cd nlp-fall-2025
```

2. **Install dependencies**

For the Streamlit app:
```bash
cd app-streamlit
pip install -r requirements-nlp.txt
```

For NLP models, you'll also need:
```bash
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
```

3. **Download NLTK data** (if needed)
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## 🚀 Usage

### Running the Web Application

```bash
cd app-streamlit
streamlit run Home.py
```

The app will open in your browser at `http://localhost:8501`

### Scraping LinkedIn Jobs

**⚠️ Important: LinkedIn Cookie Required**

You need to obtain your `li_at` cookie from LinkedIn:

1. Log into LinkedIn in Chrome
2. Open Developer Tools (F12)
3. Go to Application → Cookies → `https://www.linkedin.com`
4. Find and copy the `li_at` cookie value

Then run:
```bash
LI_AT_COOKIE="your_cookie_here" python linkedin.py
```

### Running NLP Analysis

Each NLP technique has dedicated Jupyter notebooks:

**Named Entity Recognition:**
```bash
jupyter notebook workspace/NER/NER.ipynb
```

**Topic Modeling:**
```bash
jupyter notebook workspace/Topic\ Modeling/TopicModeling_LDA.ipynb
jupyter notebook workspace/Topic\ Modeling/TopicModeling_LSA.ipynb
```

**Word Embeddings:**
```bash
jupyter notebook workspace/Word\ Embedding/Word\ Embedding_Word2Vector_UseDedup.ipynb
jupyter notebook workspace/Word\ Embedding/Word_Embedding_SBERT_UseDedup.ipynb
```

## 📊 Features Overview

### LinkedIn Job Scraper

Scrapes job listings with filters for:
- Full-time and internship positions
- Remote work options
- Mid to senior experience level
- $100K+ base salary
- Posted within the last month

Output CSV includes:
- Job Title
- Company
- Company Link
- Date
- Date Text
- Job Link
- Insights
- Description Length
- Description

### NLP Analytics

**1. Named Entity Recognition (NER)**
- Extract skills and technologies
- Identify qualifications and certifications
- Recognize company names and locations
- Custom entity types for job-specific information

**2. Topic Modeling**
- LDA (Latent Dirichlet Allocation)
- LSA (Latent Semantic Analysis)
- Discover hidden themes in job descriptions
- Visualize topic distributions

**3. Word Embeddings**
- Word2Vec for word-level semantics
- Sentence-BERT for document similarity
- Find similar jobs
- Skill relationship analysis

**4. Resume Matching**
- Extract skills from resumes
- Compute similarity scores
- Rank jobs by compatibility
- Provide match explanations

## 🔧 Configuration

### Job Titles to Scrape

Edit `linkedin.py` to customize the job titles (lines 18-28):

```python
job_titles = [
    'Data Scientist', 
    'Machine Learning Engineer',
    'Software Engineer',
    # Add your desired titles...
]
```

### Scraping Parameters

In `linkedin.py`, adjust filters:
- Experience level
- Work location (remote/hybrid/onsite)
- Salary range
- Time posted
- Job type (full-time/internship)

## 📈 Data Pipeline

1. **Collection**: LinkedIn scraper → CSV files in `scraps/`
2. **Preprocessing**: Combine and clean data → `workspace/Data/`
3. **Analysis**: NLP models process descriptions → Extract insights
4. **Visualization**: Streamlit app displays results

## 🤝 Team

Capstone Fall 2025 Team Members
- [Team information from proposal]

## 📝 Notes

- The scraper uses random delays (60-240 seconds) between job title searches to avoid rate limiting
- All scraped data is timestamped: `linkedin_jobs_YYYYMMDD_HHMMSS.csv`
- Keep your LinkedIn cookie secure and don't commit it to version control
- Runs in headless mode by default for efficiency

## 🔒 Privacy & Ethics

- Only scrapes publicly available job postings
- Respects LinkedIn's rate limits with delays
- No personal data collection
- For educational/research purposes only

## 📚 Resources

- [spaCy Documentation](https://spacy.io/)
- [Gensim Topic Modeling](https://radimrehurek.com/gensim/)
- [Sentence-BERT](https://www.sbert.net/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🐛 Troubleshooting

**Issue**: Scraper fails to load jobs
- **Solution**: Update your `li_at` cookie (cookies expire periodically)

**Issue**: NER model not found
- **Solution**: Download spaCy models: `python -m spacy download en_core_web_sm`

**Issue**: Memory errors with large datasets
- **Solution**: Process data in chunks or use sampling in the notebooks

## 📄 License

[Add your license here]

## LinkedIn Job Scraper

This project contains a LinkedIn job scraper that collects job postings based on various job titles and saves them to CSV files.

### Prerequisites

- Python 3.7+
- Chrome browser installed
- LinkedIn account

### Installation

1. Install the required package:
```bash
pip install linkedin-jobs-scraper
```

### Setup - Important: LinkedIn Cookie Required

**You MUST obtain the `li_at` cookie from your LinkedIn account for the scraper to work.**

#### How to get your `li_at` cookie:

1. Open Chrome (or your preferred browser) and log into your LinkedIn account
2. Open Developer Tools (F12 or Right-click → Inspect)
3. Go to the **Application** tab (Chrome) or **Storage** tab (Firefox)
4. In the left sidebar, expand **Cookies** and click on `https://www.linkedin.com`
5. Find the cookie named `li_at`
6. Copy the **Value** of the `li_at` cookie (it will be a long string)

![LinkedIn Cookie Location](/scraps/linkedin_cookie.png)

**Note:** Keep this cookie value private and secure. Do not share it or commit it to version control.

#### Configure the cookie in your code:

You'll need to pass the `li_at` cookie to the scraper. This is typically done by setting it in the Chrome options when initializing the scraper.

### Job Titles to be Scraped

**⚠️ Important: Update the job titles list based on your needs!**

Before running the script, you should modify the `job_titles` list in `linkedin.py` (lines 18-28) to include the job titles you want to scrape. The current list includes the following as an example:

- Digital Marketing Specialist, Business Development Manager
- Quality Assurance Analyst, Systems Administrator, Database Administrator, Cybersecurity Analyst, DevOps Engineer
- Mobile App Developer, Cloud Solutions Architect, Technical Support Engineer, SEO Specialist, Social Media Manager
- Content Marketing Manager, E-commerce Manager, Brand Manager, Public Relations Specialist, Event Coordinator
- Logistics Manager, Supply Chain Analyst, Operations Analyst, Risk Manager, Compliance Officer, Auditor, Tax Specialist
- Investment Analyst, Portfolio Manager, Real Estate Agent, Insurance Underwriter, Claims Adjuster, Actuary, Loan Officer, Credit Analyst, Treasury Analyst, Financial Planner
- Marketing Analyst, Market Research Analyst, Advertising Manager, Media Planner, Copywriter, Video Producer, Animator, Illustrator, Interior Designer, Architect
- Civil Engineer, Mechanical Engineer, Electrical Engineer, Chemical Engineer, Environmental Engineer, Biomedical Engineer, Industrial Engineer, Aerospace Engineer, Petroleum Engineer, Nuclear Engineer
- Pharmacist, Nurse Practitioner, Physician Assistant, Medical Laboratory Technician, Radiologic Technologist, Physical Therapist, Occupational Therapist, Speech-Language Pathologist, Dietitian, Respiratory Therapist
- Teacher, School Counselor, Librarian, Social Worker, Psychologist, Counselor, Therapist, Coach, Trainer, Recruiter

**Total: 70 different job titles**

**To customize:** Open `linkedin.py` and edit the `job_titles` list to include only the roles you're interested in. You can add or remove job titles as needed.

The script will scrape each job title sequentially with a random delay (60-240 seconds) between each title to avoid rate limiting.

### Usage

Run the scraper:
```bash

LI_AT_COOKIE="please put your li_at cookie here" python linkedin.py

```

### What it does

- Scrapes job listings for 70+ different job titles
- Filters for:
  - Full-time and internship positions
  - Remote work options
  - Mid to senior experience level
  - $100K+ base salary
  - Posted within the last month
- Saves each scrape session to a timestamped CSV file in the format: `linkedin_jobs_YYYYMMDD_HHMMSS.csv`
- Implements delays between requests to avoid rate limiting

### Output

Each CSV file contains the following columns:
- Job Title
- Company
- Company Link
- Date
- Date Text
- Job Link
- Insights
- Description Length
- Description

### Important Notes

- The scraper uses a 2-second delay between requests to avoid being rate-limited
- Runs in headless mode by default
- A random wait time (60-240 seconds) is added between different job title searches
- All scraped data is saved in the `scraps/` directory
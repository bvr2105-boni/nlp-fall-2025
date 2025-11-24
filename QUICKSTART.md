# Quick Start Guide - LinkedIn Job Analysis Platform

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies

```bash
cd app-streamlit
pip install -r requirements-nlp.txt
```

Download spaCy models:
```bash
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
```

### Step 2: Launch the Application

```bash
streamlit run Home.py
```

The app will open at `http://localhost:8501`

### Step 3: Load Your Data

1. Navigate to **EDA** page using the sidebar or button
2. Click **"Load Job Data"** button
3. Wait for data to load (should take a few seconds)

The system will automatically search for data in:
- `workspace/Data/Jobs_data.csv`
- `workspace/Data/combined_data.json`
- `scraps/` directory (most recent CSV)

### Step 4: Explore the Features

#### 📊 Exploratory Data Analysis
- View job market statistics
- Analyze top companies and job titles
- Explore job description patterns
- Visualize posting trends

#### 🤖 NLP Analytics
- **Named Entity Recognition**: Extract skills and qualifications
- **Topic Modeling**: Discover themes with LDA/LSA
- **Word Embeddings**: Find similar jobs
- **Resume Matching**: Upload resume and find matches

## 📋 Common Tasks

### Task 1: Analyze Job Market Trends

1. Go to **EDA** → **Job Market Overview**
2. View the timeline chart for posting trends
3. Check top job titles bar chart
4. Review description length statistics

### Task 2: Extract Skills from Job Descriptions

1. Go to **NLP Analytics** → **Named Entity Recognition**
2. Load job data if not already loaded
3. Select NER model (recommend `en_core_web_lg`)
4. Set sample size (start with 100)
5. Click **"Run NER Analysis"**
6. View extracted skills and entities

### Task 3: Discover Job Description Topics

1. Go to **NLP Analytics** → **Topic Modeling**
2. Select method (LDA or LSA)
3. Set number of topics (recommend 10)
4. Click **"Run Topic Modeling"**
5. Review topic keywords and distributions

### Task 4: Find Similar Jobs

1. Go to **NLP Analytics** → **Word Embeddings**
2. Select a job title from dropdown
3. Choose embedding method (Word2Vec or SBERT)
4. Click **"Find Similar Jobs"**
5. View ranked list of similar positions

### Task 5: Match Resume to Jobs

1. Go to **NLP Analytics** → **Resume Matching**
2. Upload your resume (PDF or TXT)
3. Set number of top matches (e.g., 10)
4. Set minimum similarity threshold (e.g., 50%)
5. Click **"Find Matching Jobs"**
6. Review ranked matches with explanations

## 🛠️ Troubleshooting

### Issue: "Could not find job data"
**Solution**: Ensure you have data files in one of these locations:
- `workspace/Data/Jobs_data.csv`
- `workspace/Data/combined_data.json`
- `scraps/*.csv`

If not, run the LinkedIn scraper first:
```bash
LI_AT_COOKIE="your_cookie" python linkedin.py
```

### Issue: "ModuleNotFoundError: No module named 'spacy'"
**Solution**: Install requirements:
```bash
pip install -r requirements-nlp.txt
python -m spacy download en_core_web_sm
```

### Issue: "Memory Error"
**Solution**: Reduce sample size in analysis settings or process data in chunks

### Issue: Page doesn't load
**Solution**: 
1. Check terminal for error messages
2. Ensure all dependencies are installed
3. Try refreshing the browser
4. Restart Streamlit: `Ctrl+C` then `streamlit run Home.py`

## 📁 Data Format

Your job data CSV should include these columns:
- `Job Title`: Position name
- `Company`: Company name
- `Company Link`: LinkedIn company URL
- `Date`: Posting date
- `Job Link`: LinkedIn job URL
- `Insights`: Additional info (location, remote, etc.)
- `Description`: Full job description text
- `Description Length`: Character count

## 🎯 Tips for Best Results

### For NER:
- Use `en_core_web_lg` for better accuracy
- Process 100-500 jobs at a time
- Review and refine entity categories

### For Topic Modeling:
- Start with 10 topics
- Increase for more granular themes
- Use LDA for interpretable topics
- Use LSA for faster processing

### For Word Embeddings:
- SBERT works better for job matching
- Word2Vec is faster for large datasets
- Cache embeddings for repeated searches

### For Resume Matching:
- Ensure resume is text-readable (not scanned image)
- Set realistic similarity threshold (50-70%)
- Review top 10-20 matches

## 🔧 Advanced Configuration

### Custom Job Titles to Scrape

Edit `linkedin.py`:
```python
job_titles = [
    'Data Scientist',
    'Machine Learning Engineer',
    # Add your titles here
]
```

### Adjust Scraping Filters

In `linkedin.py`, modify:
- Experience level
- Salary range
- Work location (remote/hybrid)
- Time posted
- Job type

### Custom Keywords for Analysis

In EDA page, customize keyword list:
```python
keywords = [
    'python', 'java', 'sql',
    # Add your keywords
]
```

## 📊 Sample Workflow

1. **Day 1**: Scrape job data
   ```bash
   LI_AT_COOKIE="cookie" python linkedin.py
   ```

2. **Day 2**: Explore data in Streamlit
   - Load data
   - View statistics
   - Identify trends

3. **Day 3**: Run NLP analysis
   - Extract entities with NER
   - Discover topics with LDA
   - Find similar jobs

4. **Day 4**: Resume matching
   - Upload test resumes
   - Analyze match scores
   - Refine matching algorithm

5. **Day 5**: Export insights
   - Save visualizations
   - Document findings
   - Share with team

## 🎓 Learning Resources

### NLP Concepts:
- [spaCy Documentation](https://spacy.io/)
- [Gensim Topic Modeling Guide](https://radimrehurek.com/gensim/auto_examples/index.html)
- [Sentence-BERT Paper](https://arxiv.org/abs/1908.10084)

### Streamlit:
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Gallery](https://streamlit.io/gallery)

### Data Visualization:
- [Plotly Documentation](https://plotly.com/python/)

## 💡 Pro Tips

1. **Cache embeddings**: Once computed, save embeddings to avoid recomputation
2. **Use sampling**: Test on small samples before full dataset
3. **Monitor memory**: Close unused apps when processing large datasets
4. **Regular updates**: Re-scrape weekly to keep data current
5. **Backup data**: Keep copies of original scraped CSVs

## 🤝 Need Help?

- Check `UPDATE_SUMMARY.md` for detailed changes
- Review Jupyter notebooks in `workspace/` for examples
- See `README.md` for full documentation
- Check error messages in terminal

## ✅ Next Steps

After getting comfortable with the basics:
1. Customize NLP models for your use case
2. Add custom entity types for NER
3. Train domain-specific Word2Vec models
4. Implement advanced filtering
5. Add export functionality for results

---

**Happy Analyzing! 🎉**

For issues or questions, refer to the main `README.md` or check the project documentation.

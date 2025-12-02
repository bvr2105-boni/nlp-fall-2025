import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import json
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from components.header import render_header

from utils import initialize_workspace

# Initialize workspace path and imports
initialize_workspace()

# Page configuration
st.set_page_config(
    page_title="EDA - LinkedIn Job Analysis",
    page_icon="📊",
    layout="wide"
)

# Load global CSS
try:
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "styles", "app.css")
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except Exception:
    pass

st.title("Exploratory Data Analysis")

# Initialize session state for data
if 'jobs_df' not in st.session_state:
    st.session_state.jobs_df = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Function to load job data
@st.cache_data
def load_job_data():
    """Load job data from workspace"""
    workspace_path = st.session_state.get('workspace_path')
    if workspace_path:
        data_path = os.path.join(workspace_path, "Data", "Jobs_data.csv")
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            return df
        else:
            # Try loading from cleaned_data.json (new cleaned data)
            json_path = os.path.join(workspace_path, "Data", "cleaned_data.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
                return df
            else:
                # Try loading from combined_data.json
                json_path = os.path.join(workspace_path, "Data", "combined_data.json")
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    df = pd.DataFrame(data)
                    return df
    return None

# Add some spacing and better intro
st.markdown("""
<div style="margin-bottom: 2rem;">
    <p style="color: #6b7280; font-size: 1.1rem; line-height: 1.6;">
        Explore comprehensive insights from LinkedIn job postings. Analyze market trends, company distributions,
        location patterns, and skill requirements to make data-driven career decisions.
    </p>
</div>
""", unsafe_allow_html=True)

# Tabs for different analyses
tab1, tab2, tab3, tab4 = st.tabs([
    "Load Data",
    "Market Overview",
    "Companies & Locations",
    "Skills Analysis"
])

with tab1:
    st.markdown("### Load Job Dataset")
    st.markdown("Import your LinkedIn job postings data to begin analysis.")

    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("Load Job Data", type="primary"):
            with st.spinner("Loading job data..."):
                try:
                    df = load_job_data()
                    if df is not None:
                        st.session_state.jobs_df = df
                        st.session_state.data_loaded = True
                        st.success(f"✅ Successfully loaded {len(df):,} job postings")
                        st.rerun()
                    else:
                        st.error("❌ Could not find job data. Please ensure data files exist in workspace/Data/")
                except Exception as e:
                    st.error(f"❌ Error loading data: {e}")

    with col2:
        if st.session_state.data_loaded and st.session_state.jobs_df is not None:
            df = st.session_state.jobs_df
            st.success(f"✅ Dataset ready: {len(df):,} job postings loaded")

    if st.session_state.data_loaded and st.session_state.jobs_df is not None:
        df = st.session_state.jobs_df

        # Key metrics in a nice grid
        st.markdown("### Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Jobs", f"{len(df):,}")
        with col2:
            if 'Company' in df.columns:
                st.metric("Companies", f"{df['Company'].nunique():,}")
        with col3:
            if 'Job Title' in df.columns:
                st.metric("Job Titles", f"{df['Job Title'].nunique():,}")
        with col4:
            if 'Description' in df.columns:
                avg_length = df['Description'].str.len().mean()
                st.metric("Avg Description", f"{avg_length:.0f} chars")

        # Sample data preview
        st.markdown("### Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Column information
        with st.expander("📋 Column Information"):
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum(),
                'Unique Values': df.nunique()
            })
            st.dataframe(col_info, use_container_width=True)
            
        # Data quality summary
        with st.expander("🔍 Data Quality Summary"):
            st.markdown("#### Missing Values Overview")
            missing_data = df.isnull().sum()
            missing_percent = (missing_data / len(df)) * 100
            missing_df = pd.DataFrame({
                'Missing Count': missing_data,
                'Missing %': missing_percent.round(2)
            })
            st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
            
            st.markdown("#### Duplicate Rows")
            duplicates = df.duplicated().sum()
            st.metric("Duplicate Rows", duplicates)

with tab2:
    st.markdown("### Job Market Overview")
    
    if not st.session_state.data_loaded:
        st.info("📌 Please load the dataset first from the 'Load Dataset' tab.")
    else:
        df = st.session_state.jobs_df
        
        # Job posting trends over time
        if 'Date' in df.columns or 'Date Text' in df.columns:
            st.markdown("#### Job Posting Timeline")
            date_col = 'Date' if 'Date' in df.columns else 'Date Text'
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                date_counts = df[date_col].value_counts().sort_index()
                
                fig = px.line(
                    x=date_counts.index, 
                    y=date_counts.values,
                    labels={'x': 'Date', 'y': 'Number of Job Postings'},
                    title='Job Postings Over Time'
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not parse dates: {e}")
        
        # Top job titles
        if 'Job Title' in df.columns:
            st.markdown("#### Top Job Titles")
            top_titles = df['Job Title'].value_counts().head(20)
            
            fig = px.bar(
                x=top_titles.values,
                y=top_titles.index,
                orientation='h',
                labels={'x': 'Number of Postings', 'y': 'Job Title'},
                title='Top 20 Job Titles'
            )
            fig.update_traces(hoverinfo='none')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        # Description length analysis
        if 'Description Length' in df.columns or 'Description' in df.columns:
            st.markdown("#### Job Description Analysis")
            if 'Description Length' not in df.columns and 'Description' in df.columns:
                df['Description Length'] = df['Description'].str.len()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Description Length", f"{df['Description Length'].mean():.0f} chars")
            with col2:
                st.metric("Median Description Length", f"{df['Description Length'].median():.0f} chars")
            with col3:
                st.metric("Max Description Length", f"{df['Description Length'].max():.0f} chars")
            
            fig = px.histogram(
                df, 
                x='Description Length',
                nbins=50,
                title='Distribution of Job Description Lengths'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Word count analysis
            if 'Description' in df.columns:
                df['Word Count'] = df['Description'].str.split().str.len()
                fig2 = px.histogram(
                    df,
                    x='Word Count',
                    nbins=50,
                    title='Distribution of Word Counts in Descriptions'
                )
                st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown("### Company & Location Analysis")
    
    if not st.session_state.data_loaded:
        st.info("📌 Please load the dataset first from the 'Load Dataset' tab.")
    else:
        df = st.session_state.jobs_df
        
        # Top companies
        if 'Company' in df.columns:
            st.markdown("#### Top Companies by Job Postings")
            top_companies = df['Company'].value_counts().head(20)
            
            fig = px.bar(
                x=top_companies.values,
                y=top_companies.index,
                orientation='h',
                labels={'x': 'Number of Job Postings', 'y': 'Company'},
                title='Top 20 Companies'
            )
            fig.update_traces(hoverinfo='none')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        # Insights analysis (location, work type, etc.)
        if 'Insights' in df.columns:
            st.markdown("#### Job Insights Distribution")
            
            # Extract insights
            all_insights = []
            for insights in df['Insights'].dropna():
                if isinstance(insights, str):
                    all_insights.extend(insights.split(','))
            
            if all_insights:
                insight_counts = Counter([i.strip() for i in all_insights])
                insight_df = pd.DataFrame.from_dict(insight_counts, orient='index', columns=['Count'])
                insight_df = insight_df.sort_values('Count', ascending=False).head(50)
                
                fig = px.bar(
                    insight_df,
                    x=insight_df.index,
                    y='Count',
                    title='Top Job Insights',
                    labels={'x': 'Insight', 'y': 'Count'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
        
        # Location analysis
        if 'Location' in df.columns:
            st.markdown("#### Top Locations")
            top_locations = df['Location'].value_counts().head(15)
            
            fig = px.bar(
                x=top_locations.values,
                y=top_locations.index,
                orientation='h',
                labels={'x': 'Number of Job Postings', 'y': 'Location'},
                title='Top 15 Locations'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown("### Skills & Requirements Analysis")
    
    if not st.session_state.data_loaded:
        st.info("📌 Please load the dataset first from the 'Load Dataset' tab.")
    else:
        df = st.session_state.jobs_df
        
        st.markdown("""
        This section analyzes skills and requirements from job descriptions using text analysis.
        
        **Available NLP Features:**
        - Named Entity Recognition (NER) - Extract skills, technologies, and qualifications
        - Topic Modeling - Discover common themes in job descriptions
        - Word Embeddings - Find similar jobs and skill relationships
        
        For detailed NLP analysis, please visit the **NLP Analytics** page.
        """)
        
        # Simple keyword analysis
        if 'Description' in df.columns:
            st.markdown("#### Common Keywords in Job Descriptions")
            
            # Common tech keywords to search for
            keywords = [
                'python', 'java', 'javascript', 'sql', 'aws', 'azure', 'docker', 
                'kubernetes', 'machine learning', 'data science', 'ai', 'nlp',
                'react', 'node.js', 'typescript', 'git', 'agile', 'scrum',
                'bachelor', 'master', 'phd', 'remote', 'hybrid', 'onsite',
                'senior', 'junior', 'lead', 'manager', 'director'
            ]
            
            keyword_counts = {}
            for keyword in keywords:
                count = df['Description'].str.lower().str.contains(keyword, na=False).sum()
                if count > 0:
                    keyword_counts[keyword] = count
            
            if keyword_counts:
                keyword_df = pd.DataFrame.from_dict(
                    keyword_counts, 
                    orient='index', 
                    columns=['Count']
                ).sort_values('Count', ascending=False)
                
                fig = px.bar(
                    keyword_df.head(50),
                    x=keyword_df.head(50).index,
                    y='Count',
                    title='Top Keywords in Job Descriptions',
                    labels={'x': 'Keyword', 'y': 'Number of Mentions'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Experience level analysis
                st.markdown("#### Experience Level Distribution")
                experience_keywords = {
                    'Entry Level': ['junior', 'entry', 'graduate', '0-2 years', '1-3 years'],
                    'Mid Level': ['mid', '3-5 years', '4-6 years', 'intermediate'],
                    'Senior Level': ['senior', 'lead', '7+ years', '8+ years', '10+ years'],
                    'Executive': ['director', 'vp', 'chief', 'executive', 'manager']
                }
                
                experience_counts = {}
                for level, terms in experience_keywords.items():
                    count = sum(df['Description'].str.lower().str.contains(term, na=False).sum() for term in terms)
                    if count > 0:
                        experience_counts[level] = count
                
                if experience_counts:
                    exp_df = pd.DataFrame.from_dict(experience_counts, orient='index', columns=['Count'])
                    fig2 = px.pie(
                        exp_df,
                        values='Count',
                        names=exp_df.index,
                        title='Experience Level Distribution'
                    )
                    st.plotly_chart(fig2, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("💡 **Tip:** Use the NLP Analytics page for advanced text analysis including NER, Topic Modeling, and Word Embeddings.")

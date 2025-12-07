"""
NLP analysis components for job description analysis
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

def create_keyword_analysis(df, keywords=None):
    """
    Create keyword frequency analysis visualization
    
    Args:
        df: DataFrame with job descriptions
        keywords: List of keywords to search for (optional)
    """
    if 'Description' not in df.columns:
        st.warning("No description column found")
        return
    
    if keywords is None:
        keywords = [
            'python', 'java', 'javascript', 'sql', 'aws', 'azure', 'docker',
            'kubernetes', 'machine learning', 'data science', 'ai', 'nlp',
            'react', 'node.js', 'typescript', 'git', 'agile', 'scrum',
            'bachelor', 'master', 'phd', 'remote', 'hybrid'
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
    else:
        st.info("No keywords found in descriptions")

def create_company_distribution(df, top_n=20):
    """Create company distribution visualization"""
    if 'Company' not in df.columns:
        st.warning("No company column found")
        return
    
    top_companies = df['Company'].value_counts().head(top_n)
    
    fig = px.bar(
        x=top_companies.values,
        y=top_companies.index,
        orientation='h',
        labels={'x': 'Number of Job Postings', 'y': 'Company'},
        title=f'Top {top_n} Companies by Job Postings'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

def create_job_title_distribution(df, top_n=20):
    """Create job title distribution visualization"""
    if 'Job Title' not in df.columns:
        st.warning("No job title column found")
        return
    
    top_titles = df['Job Title'].value_counts().head(top_n)
    
    fig = px.bar(
        x=top_titles.values,
        y=top_titles.index,
        orientation='h',
        labels={'x': 'Number of Postings', 'y': 'Job Title'},
        title=f'Top {top_n} Job Titles'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, use_container_width=True)

def create_description_length_analysis(df):
    """Analyze and visualize job description lengths"""
    if 'Description' not in df.columns:
        st.warning("No description column found")
        return
    
    if 'Description Length' not in df.columns:
        df['Description Length'] = df['Description'].str.len()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Length", f"{df['Description Length'].mean():.0f} chars")
    with col2:
        st.metric("Median Length", f"{df['Description Length'].median():.0f} chars")
    with col3:
        st.metric("Max Length", f"{df['Description Length'].max():.0f} chars")
    
    fig = px.histogram(
        df,
        x='Description Length',
        nbins=50,
        title='Distribution of Job Description Lengths',
        labels={'Description Length': 'Length (characters)'}
    )
    st.plotly_chart(fig, use_container_width=True)

def create_insights_analysis(df):
    """Analyze job insights (location, work type, etc.)"""
    if 'Insights' not in df.columns:
        st.warning("No insights column found")
        return
    
    all_insights = []
    for insights in df['Insights'].dropna():
        if isinstance(insights, str):
            all_insights.extend([i.strip() for i in insights.split(',')])
    
    if all_insights:
        insight_counts = Counter(all_insights)
        insight_df = pd.DataFrame.from_dict(
            insight_counts,
            orient='index',
            columns=['Count']
        ).sort_values('Count', ascending=False).head(15)
        
        fig = px.bar(
            insight_df,
            x=insight_df.index,
            y='Count',
            title='Top Job Insights',
            labels={'x': 'Insight', 'y': 'Count'}
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No insights data available")

def create_timeline_plot(df):
    """Create timeline visualization of job postings"""
    date_col = 'Date' if 'Date' in df.columns else 'Date Text'
    
    if date_col not in df.columns:
        st.warning("No date column found")
        return
    
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

def display_job_metrics(df):
    """Display key metrics about the job dataset"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Jobs", f"{len(df):,}")
    
    with col2:
        if 'Company' in df.columns:
            st.metric("Unique Companies", f"{df['Company'].nunique():,}")
    
    with col3:
        if 'Job Title' in df.columns:
            st.metric("Unique Titles", f"{df['Job Title'].nunique():,}")
    
    with col4:
        if 'Description Length' in df.columns or 'Description' in df.columns:
            if 'Description Length' not in df.columns:
                df['Description Length'] = df['Description'].str.len()
            st.metric("Avg Description", f"{df['Description Length'].mean():.0f} chars")

"""
Database and data loading functions for LinkedIn Job Analysis
"""
import pandas as pd
import os
import json

# Columns to use for the NLP job analysis
SELECTED_COLUMNS = [
    'Job Title',
    'Company',
    'Company Link',
    'Date',
    'Date Text',
    'Job Link',
    'Insights',
    'Description Length',
    'Description'
]

def load_job_data(workspace_path=None):
    """
    Load job data from CSV or JSON file
    
    Args:
        workspace_path: Path to the workspace directory
    
    Returns:
        pandas DataFrame with job data
    """
    if workspace_path is None:
        workspace_path = os.path.join(os.path.dirname(__file__), '..', '..', 'workspace')
    
    # Try loading from CSV
    csv_path = os.path.join(workspace_path, "Data", "Jobs_data.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df
    
    # Try loading from JSON
    json_path = os.path.join(workspace_path, "Data", "combined_data.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        return df
    
    # Try loading from scraps directory
    scraps_path = os.path.join(os.path.dirname(workspace_path), "scraps")
    if os.path.exists(scraps_path):
        csv_files = [f for f in os.listdir(scraps_path) if f.endswith('.csv')]
        if csv_files:
            # Load the most recent file
            latest_file = sorted(csv_files)[-1]
            df = pd.read_csv(os.path.join(scraps_path, latest_file))
            return df
    
    raise FileNotFoundError("Could not find job data files")

def get_job_by_id(df, job_id):
    """Get a specific job by index"""
    if job_id < len(df):
        return df.iloc[job_id]
    return None

def search_jobs(df, query, search_columns=['Job Title', 'Company', 'Description']):
    """
    Search jobs by query string
    
    Args:
        df: DataFrame with job data
        query: Search query
        search_columns: Columns to search in
    
    Returns:
        Filtered DataFrame
    """
    if not query:
        return df
    
    query_lower = query.lower()
    mask = False
    
    for col in search_columns:
        if col in df.columns:
            mask = mask | df[col].str.lower().str.contains(query_lower, na=False)
    
    return df[mask]

def filter_by_company(df, company):
    """Filter jobs by company"""
    if 'Company' in df.columns:
        return df[df['Company'] == company]
    return df

def filter_by_date_range(df, start_date, end_date):
    """Filter jobs by date range"""
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        mask = (df['Date'] >= start_date) & (df['Date'] <= end_date)
        return df[mask]
    return df

def get_top_companies(df, n=10):
    """Get top N companies by number of job postings"""
    if 'Company' in df.columns:
        return df['Company'].value_counts().head(n)
    return pd.Series()

def get_top_job_titles(df, n=10):
    """Get top N job titles by frequency"""
    if 'Job Title' in df.columns:
        return df['Job Title'].value_counts().head(n)
    return pd.Series()

def get_stats(df):
    """Get basic statistics about the dataset"""
    stats = {
        'total_jobs': len(df),
        'unique_companies': df['Company'].nunique() if 'Company' in df.columns else 0,
        'unique_titles': df['Job Title'].nunique() if 'Job Title' in df.columns else 0,
        'avg_description_length': df['Description Length'].mean() if 'Description Length' in df.columns else 0,
    }
    return stats

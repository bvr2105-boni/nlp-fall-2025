import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import json

from utils import initialize_workspace

# Initialize workspace path and imports
initialize_workspace()

# Page configuration
st.set_page_config(
    page_title="Import Embeddings - Database",
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

st.title("📊 Import Word2Vec Embeddings to Database")

# Try to import NLP libraries
try:
    from gensim.models import Word2Vec
    from gensim.utils import simple_preprocess
    GENSIM_AVAILABLE = True
except ImportError:
    GENSIM_AVAILABLE = False

# Try to import SBERT utilities
try:
    from functions.nlp_models import load_sbert_model, compute_job_embeddings_sbert
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

# Helper functions
def simple_tokenize(text):
    """Simple tokenization"""
    if pd.isna(text):
        return []
    return str(text).split()

def get_doc_embedding_w2v(tokens, model):
    """Get document embedding using Word2Vec"""
    if not GENSIM_AVAILABLE or model is None:
        # Get vector size from model if available, otherwise default to 300
        vector_size = getattr(model, 'vector_size', 300) if model else 300
        return np.zeros(vector_size, dtype="float32")
    
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        # Use model's vector_size, default to 300 for Word2Vec
        vector_size = getattr(model, 'vector_size', 300)
        return np.zeros(vector_size, dtype="float32")
    return np.mean(vectors, axis=0)

@st.cache_data
def compute_job_embeddings_w2v(job_texts, _model):
    """Compute embeddings for jobs using Word2Vec"""
    embeddings = []
    for text in job_texts:
        tokens = simple_tokenize(text)
        emb = get_doc_embedding_w2v(tokens, _model)
        embeddings.append(emb)
    return np.array(embeddings)

def load_job_data():
    """Load job data from workspace"""
    workspace_path = st.session_state.get('workspace_path')
    if workspace_path:
        # Try loading from cleaned_data.json first (default for NLP)
        json_path = os.path.join(workspace_path, "Data", "cleaned_data.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            return df
        
        # Fallback to combined_data.json (has job links)
        json_path = os.path.join(workspace_path, "Data", "combined_data.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
            return df
        
        # Fallback to CSV
        csv_path = os.path.join(workspace_path, "Data_Cleaning", "cleaned_job_data.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            return df
    
    # Try loading from current directory
    csv_path = "workspace/Data_Cleaning/cleaned_job_data.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df
    
    return None

# Initialize session state
if 'cleaned_jobs_df' not in st.session_state:
    st.session_state.cleaned_jobs_df = None

if 'w2v_model' not in st.session_state:
    st.session_state.w2v_model = None

# Main content
st.markdown("""
### Overview
Import computed embeddings to PostgreSQL database with pgvector extension:
- Computes **Word2Vec** embeddings for all jobs
- Computes **SBERT** embeddings for all jobs (if available)
- Stores embeddings in `embedding` (SBERT, 384 dimensions) and `word2vec_embedding` (Word2Vec, 300 dimensions)
- Enables fast vector similarity search using pgvector indexes
""")

# Check database availability
try:
    from functions.database import (
        create_db_engine, setup_jobs_table,
        insert_job_with_multiple_embeddings,
        batch_insert_jobs_with_embeddings,
        drop_jobs_table,
        reset_jobs_table,
        backup_jobs_to_sql,
        restore_jobs_from_latest_sql_backup,
    )
    DB_AVAILABLE = True
except ImportError as e:
    DB_AVAILABLE = False
    st.error(f"Database functions not available: {e}")
    st.info("Please ensure database connection is configured in `.env` or `docker-compose.yml`")

if not DB_AVAILABLE:
    st.stop()

# Database Setup Section
st.markdown("---")
st.markdown("### 🗄️ Database Setup")

col_setup, col_backup, col_reset = st.columns(3)

with col_setup:
    if st.button("Setup / Ensure Jobs Table", key="setup_db_table", type="primary"):
        with st.spinner("Setting up database table..."):
            if setup_jobs_table():
                # After ensuring schema, try to restore from latest SQL backup if available
                restored = restore_jobs_from_latest_sql_backup()
                if restored:
                    st.success("✅ Database table created/updated and restored from latest SQL backup!")
                else:
                    st.success("✅ Database table created/updated! (No SQL backup restored; either none found or restore failed — check logs/backups.)")
            else:
                st.error("❌ Failed to setup database table. Check database connection.")

with col_backup:

    if st.button("Backup Jobs to SQL", key="backup_jobs_sql"):
        with st.spinner("Exporting jobs table as SQL..."):
            backup_sql_path = backup_jobs_to_sql()
            if backup_sql_path:
                st.success(f"✅ Jobs exported as SQL to `{backup_sql_path}`")
                st.info("You can restore data by running this SQL file against a database where the `jobs` table already exists.")
            else:
                st.error("❌ Failed to export jobs to SQL. Check logs and permissions.")

with col_reset:
    if st.button("Delete & Recreate Jobs Table", key="reset_db_table"):
        with st.spinner("Dropping and recreating jobs table..."):
            if reset_jobs_table():
                st.success("✅ Jobs table dropped and recreated successfully!")
            else:
                st.error("❌ Failed to reset jobs table. Check database connection and permissions.")

# Load Job Data Section
st.markdown("---")
st.markdown("### 📁 Load Job Data")

if st.session_state.cleaned_jobs_df is None:
    if st.button("Load Job Data", key="db_import_load_jobs", type="primary"):
        with st.spinner("Loading job data..."):
            df = load_job_data()
            if df is not None:
                st.session_state.cleaned_jobs_df = df
                st.success(f"✅ Loaded {len(df):,} job postings")
                st.rerun()
            else:
                st.error("❌ Could not load data. Please check that job data files exist in the workspace.")
else:
    df = st.session_state.cleaned_jobs_df
    st.success(f"✅ Working with {len(df):,} job postings")
    
    # Show data preview
    with st.expander("📋 Preview Job Data"):
        st.dataframe(df.head(10))
        st.write(f"**Columns**: {', '.join(df.columns.tolist())}")

# Word2Vec Model Section
st.markdown("---")
st.markdown("### 🤖 Word2Vec Model")

if st.session_state.w2v_model is None:
    st.warning("⚠️ Word2Vec model not loaded.")
    
    # Option to load from file
    st.markdown("#### Load Word2Vec Model")
    
    workspace_path = st.session_state.get('workspace_path')
    if workspace_path:
        models_dir = os.path.join(workspace_path, "models")
        if os.path.exists(models_dir):
            w2v_files = [f for f in os.listdir(models_dir) if f == 'word2vec_model.joblib']
            if w2v_files:
                if st.button("Load Saved Word2Vec Model", key="load_w2v_model"):
                    try:
                        import joblib
                        model_path = os.path.join(models_dir, w2v_files[0])
                        saved_data = joblib.load(model_path)
                        st.session_state.w2v_model = saved_data['model']
                        st.success("✅ Word2Vec model loaded!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to load model: {e}")
            else:
                st.info("No saved Word2Vec model found. Please train a model in the NLP Analytics page first.")
        else:
            st.info("Models directory not found. Please train a Word2Vec model in the NLP Analytics page first.")
    else:
        st.info("Workspace path not set. Please train a Word2Vec model in the NLP Analytics page first.")
    
    st.info("💡 **Tip**: Train or load a Word2Vec model in the **NLP Analytics** page (Word Embeddings tab) first, then return here to import embeddings.")
else:
    st.success("✅ Word2Vec model ready")
    
    # Show model info
    model_vector_size = None
    if hasattr(st.session_state.w2v_model, 'vector_size'):
        model_vector_size = st.session_state.w2v_model.vector_size
        st.info(f"**Model Vector Size**: {model_vector_size} dimensions")
        
        # Warn if dimension doesn't match database schema
        if model_vector_size != 300:
            st.warning(f"⚠️ **Dimension Mismatch**: Your model has {model_vector_size} dimensions, but the database expects 300 dimensions.")
            st.info("💡 **Options**:")
            st.write("1. Retrain the model with `vector_size=300` in the NLP Analytics page")
            st.write("2. Or update the database schema to match your model's dimension")
            st.write("   ```sql")
            st.write(f"   ALTER TABLE jobs ALTER COLUMN word2vec_embedding TYPE vector({model_vector_size});")
            st.write("   ```")
    else:
        st.warning("⚠️ Could not determine model vector size")
    
    if hasattr(st.session_state.w2v_model, 'wv') and hasattr(st.session_state.w2v_model.wv, 'key_to_index'):
        st.info(f"**Vocabulary Size**: {len(st.session_state.w2v_model.wv.key_to_index):,} words")

# Import Embeddings Section
st.markdown("---")
st.markdown("### 📤 Import Embeddings to Database")

if st.session_state.cleaned_jobs_df is None:
    st.warning("⚠️ Please load job data first.")
elif st.session_state.w2v_model is None:
    st.warning("⚠️ Please load or train a Word2Vec model first.")
else:
    df = st.session_state.cleaned_jobs_df
    
    # Import options
    col1, col2 = st.columns(2)
    
    with col1:
        batch_size = st.number_input(
            "Batch Size for Import",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
            help="Number of jobs to process in each batch"
        )
    
    with col2:
        # Get available text columns
        text_columns = [col for col in df.columns if 'text' in col.lower() or 'description' in col.lower() or 'job' in col.lower()]
        if not text_columns:
            text_columns = df.columns.tolist()
        
        text_column = st.selectbox(
            "Text Column",
            options=text_columns,
            index=0 if 'job_text_cleaned' in text_columns else 0,
            help="Column to use for computing embeddings"
        )
    
    if st.button("Import SBERT + Word2Vec Embeddings to Database", type="primary", key="import_w2v_to_db"):
        # Get text column
        if text_column not in df.columns:
            st.error(f"Column '{text_column}' not found in dataframe. Available columns: {', '.join(df.columns)}")
        else:
            job_texts = df[text_column].dropna().tolist()
            num_jobs = len(job_texts)
            
            if num_jobs == 0:
                st.error("No job texts found in the selected column")
            else:
                progress_container = st.container()
                progress_bar = progress_container.progress(0)
                status_text = progress_container.empty()
                
                # Compute Word2Vec embeddings
                status_text.text(f"Computing Word2Vec embeddings for {num_jobs} jobs...")
                progress_bar.progress(0.1)
                
                w2v_embeddings = compute_job_embeddings_w2v(job_texts, st.session_state.w2v_model)
                
                if w2v_embeddings is None or len(w2v_embeddings) == 0:
                    st.error("Failed to compute Word2Vec embeddings")
                    st.stop()

                # Check Word2Vec embedding dimension
                w2v_dim = len(w2v_embeddings[0]) if len(w2v_embeddings) > 0 else 0
                if w2v_dim != 300:
                    st.warning(f"⚠️ **Dimension Mismatch**: Word2Vec embeddings have {w2v_dim} dimensions, but database expects 300.")
                    st.info("The import may fail. Consider updating the database schema or retraining the model.")
                    
                    if not st.checkbox("Continue anyway (may fail)", key="continue_dim_mismatch"):
                        st.stop()

                # Optionally compute SBERT embeddings
                sbert_embeddings = None
                if SBERT_AVAILABLE:
                    status_text.text(f"Computing SBERT embeddings for {num_jobs} jobs...")
                    progress_bar.progress(0.2)
                    try:
                        sbert_model = load_sbert_model()
                        if sbert_model:
                            sbert_embeddings = compute_job_embeddings_sbert(job_texts, sbert_model)
                        else:
                            st.warning("SBERT model could not be loaded; proceeding with Word2Vec only.")
                    except Exception as e:
                        st.warning(f"SBERT embedding computation failed: {e}. Proceeding with Word2Vec only.")
                else:
                    st.info("SBERT utilities not available. Only Word2Vec embeddings will be imported.")

                # If SBERT embeddings exist, validate dimension
                if sbert_embeddings is not None and len(sbert_embeddings) > 0:
                    sbert_dim = len(sbert_embeddings[0])
                    if sbert_dim != 384:
                        st.warning(f"⚠️ **Dimension Mismatch**: SBERT embeddings have {sbert_dim} dimensions, but database expects 384.")
                        if not st.checkbox("Continue anyway with SBERT (may fail)", key="continue_sbert_dim_mismatch"):
                            st.stop()
                
                progress_bar.progress(0.3)
                status_text.text(f"Preparing data for database import...")
                
                # Prepare batch data
                batch_data = []
                valid_indices = df[text_column].notna()
                valid_df = df[valid_indices].reset_index(drop=True)
                
                for idx, (_, row) in enumerate(valid_df.iterrows()):
                    if idx < len(w2v_embeddings):
                        job_id = str(row.get('id', row.get('Job Id', row.get('job_id', idx))))
                        job_text = row.get(text_column, '')
                        company = row.get('Company', row.get('company', None))
                        title = row.get('Job Title', row.get('job_title', row.get('title', None)))
                        
                        # Convert numpy arrays to lists
                        w2v_embedding = w2v_embeddings[idx].tolist() if hasattr(w2v_embeddings[idx], 'tolist') else list(w2v_embeddings[idx])
                        
                        job_record = {
                            'id': job_id,
                            'title': title,
                            'company': company,
                            'text': job_text,
                            'word2vec_embedding': w2v_embedding
                        }

                        if sbert_embeddings is not None and idx < len(sbert_embeddings):
                            sbert_embedding = sbert_embeddings[idx].tolist() if hasattr(sbert_embeddings[idx], 'tolist') else list(sbert_embeddings[idx])
                            job_record['embedding'] = sbert_embedding

                        batch_data.append(job_record)
                
                progress_bar.progress(0.5)
                status_text.text(f"Importing {len(batch_data)} jobs to database in batches of {batch_size}...")
                
                # Import in batches
                success_count = 0
                total_batches = (len(batch_data) + batch_size - 1) // batch_size
                
                for batch_idx in range(0, len(batch_data), batch_size):
                    batch = batch_data[batch_idx:batch_idx + batch_size]
                    current_batch = (batch_idx // batch_size) + 1
                    
                    progress = 0.5 + (current_batch / total_batches) * 0.5
                    progress_bar.progress(progress)
                    status_text.text(f"Importing batch {current_batch}/{total_batches} ({len(batch)} jobs)...")
                    
                    if batch_insert_jobs_with_embeddings(batch):
                        success_count += len(batch)
                    else:
                        st.warning(f"Failed to import batch {current_batch}")
                
                progress_bar.progress(1.0)
                status_text.text(f"✅ Successfully imported {success_count}/{len(batch_data)} jobs with embeddings!")
                
                st.success(f"✅ Import complete! {success_count} jobs with Word2Vec/SBERT embeddings are now in the database.")
                
                # Show summary
                with st.expander("📊 Import Summary", expanded=True):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Jobs Processed", f"{num_jobs:,}")
                    with col2:
                        st.metric("Successfully Imported", f"{success_count:,}")
                    with col3:
                        w2v_dim_display = len(w2v_embeddings[0]) if len(w2v_embeddings) > 0 else 'N/A'
                        st.metric("Word2Vec Embedding Dim", w2v_dim_display)
                        if w2v_dim_display != 300 and w2v_dim_display != 'N/A':
                            st.caption(f"⚠️ Expected 300, got {w2v_dim_display}")
                    
                    st.markdown("**Details:**")
                    st.write(f"- **Database columns**: `embedding` (SBERT, if available), `word2vec_embedding`")
                    st.write(f"- **Vector search**: Enabled with pgvector index")
                    st.write(f"- **Batch size used**: {batch_size}")
                    st.write(f"- **Text column**: `{text_column}`")

# Footer
st.markdown("---")
st.markdown("""
### 📚 Notes

- **Word2Vec Model**: Train or load a Word2Vec model in the **NLP Analytics** page (Word Embeddings tab) before importing
- **SBERT Model**: Install `sentence-transformers` and configure SBERT in the environment to store SBERT embeddings
- **Database**: Ensure PostgreSQL with pgvector extension is running and configured
- **Embeddings**: SBERT embeddings are 384‑dim, Word2Vec embeddings are 300‑dim for vector similarity search
- **Performance**: Large datasets are processed in batches for optimal performance

**Related Pages:**
- **NLP Analytics**: Train Word2Vec models and compute embeddings
- **Resume Matching**: Use imported embeddings for job matching
""")

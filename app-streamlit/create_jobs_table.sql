-- Create jobs table with vector embeddings
-- Requires pgvector extension

-- Enable pgvector extension (run as superuser)
CREATE EXTENSION IF NOT EXISTS vector;

-- Create jobs table
CREATE TABLE IF NOT EXISTS jobs (
    id VARCHAR(255) PRIMARY KEY,
    text TEXT NOT NULL,
    company VARCHAR(255),
    title VARCHAR(255),
    embedding vector(1536)  -- OpenAI ada-002 embeddings
);

-- Create index for vector similarity search
CREATE INDEX IF NOT EXISTS jobs_embedding_idx ON jobs USING ivfflat (embedding vector_cosine_ops);

-- Optional: Create index for text search
CREATE INDEX IF NOT EXISTS jobs_text_idx ON jobs USING gin(to_tsvector('english', text));
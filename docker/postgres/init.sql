CREATE EXTENSION IF NOT EXISTS vector;


CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    title VARCHAR(500),
    source VARCHAR(255),
    metadata JSONB,
    embedding vector(384),
    similarity_score FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);


CREATE TABLE IF NOT EXISTS conversation_history (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    user_query TEXT NOT NULL,
    assistant_response TEXT NOT NULL,
    rag_context JSONB,
    agent_tools_used JSONB,
    response_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);


CREATE INDEX CONCURRENTLY IF NOT EXISTS documents_content_idx ON documents USING gin(to_tsvector('english', content));
CREATE INDEX CONCURRENTLY IF NOT EXISTS documents_title_idx ON documents(title) WHERE title IS NOT NULL;
CREATE INDEX CONCURRENTLY IF NOT EXISTS documents_source_idx ON documents(source) WHERE source IS NOT NULL;


CREATE INDEX CONCURRENTLY IF NOT EXISTS documents_metadata_idx ON documents USING gin(metadata) WHERE metadata IS NOT NULL;

CREATE INDEX CONCURRENTLY IF NOT EXISTS documents_created_at_idx ON documents(created_at);
CREATE INDEX CONCURRENTLY IF NOT EXISTS documents_updated_at_idx ON documents(updated_at);

CREATE INDEX CONCURRENTLY IF NOT EXISTS conversation_history_session_idx ON conversation_history(session_id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS conversation_history_created_at_idx ON conversation_history(created_at);

CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();


GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;


ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;
ALTER SYSTEM SET effective_io_concurrency = 200;

CREATE OR REPLACE FUNCTION get_database_stats()
RETURNS TABLE(
    total_documents INTEGER,
    documents_with_embeddings INTEGER,
    avg_content_length NUMERIC,
    database_size TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        (SELECT COUNT(*)::INTEGER FROM documents) as total_documents,
        (SELECT COUNT(*)::INTEGER FROM documents WHERE embedding IS NOT NULL) as documents_with_embeddings,
        (SELECT ROUND(AVG(LENGTH(content)), 2) FROM documents) as avg_content_length,
        pg_size_pretty(pg_database_size(current_database())) as database_size;
END;
$$ LANGUAGE plpgsql;

DO $$
BEGIN
    RAISE NOTICE 'RAG microservice database initialized successfully!';
    RAISE NOTICE 'Vector extension enabled: %', (SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector'));
    RAISE NOTICE 'Sample documents inserted: %', (SELECT COUNT(*) FROM documents);
END $$;
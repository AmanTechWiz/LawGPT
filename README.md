# Legal RAG System

A Retrieval-Augmented Generation (RAG) system designed for legal document analysis and research. This system provides PDF document ingestion, vector-based similarity search, and context-aware response generation for legal queries.

## Architecture Overview

The system is built using Docker containers with the following components:

- **FastAPI Backend**: RESTful API server handling queries and document ingestion
- **PostgreSQL with pgvector**: Vector database for document storage and similarity search
- **Redis**: Caching layer for improved performance
- **Docker Compose**: Containerized deployment for development and testing
- **OpenAI Integration**: LLM-powered response generation and embedding creation

## System Components

### Core Services

1. **PDF Ingestion Service** (`app/services/pdf_ingestion.py`)
   - Extracts text from PDF documents using PyPDF2
   - Implements intelligent chunking with sentence boundary preservation
   - Generates comprehensive metadata for each document chunk
   - Handles error recovery and processing validation

2. **Lightweight LLM RAG** (`app/services/lightweight_llm_rag.py`)
   - Manages document embeddings using OpenAI's text-embedding-3-small
   - Implements hybrid search combining vector similarity and text search
   - Handles query processing and context retrieval
   - Provides response generation with source citations

3. **Legal Tools** (`app/services/legal_tools.py`)
   - Analyzes legal responses for accuracy and relevance
   - Extracts legal citations and domain classifications
   - Provides response quality assessment

4. **LangChain Agent** (`app/services/langchain_agent.py`)
   - Implements LangChain-based legal research agent
   - Provides structured legal research capabilities
   - Handles complex legal query processing

5. **Adaptive RAG Orchestrator** (`app/services/adaptive_rag_orchestrator.py`)
   - Implements intent-based query classification
   - Adapts retrieval and generation strategies based on query type
   - Provides optimized response formatting

6. **Multi-Hop Reasoning** (`app/services/multi_hop_reasoning.py`)
   - Handles complex legal queries requiring multiple reasoning steps
   - Implements chain-of-thought processing
   - Provides step-by-step reasoning validation

7. **Feedback System** (`app/services/feedback_system.py`)
   - Collects and processes user feedback
   - Provides system performance analytics
   - Implements feedback-based improvements

### Database Schema

The system uses PostgreSQL with the pgvector extension for vector operations:

```sql
-- Documents table with vector embeddings
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    title VARCHAR(500),
    source VARCHAR(255),
    metadata JSONB,
    embedding vector(1536),
    similarity_score FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Conversation history for session management
CREATE TABLE conversation_history (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) NOT NULL,
    user_query TEXT NOT NULL,
    assistant_response TEXT NOT NULL,
    rag_context JSONB,
    agent_tools_used JSONB,
    response_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

## Installation and Setup

### Prerequisites

- Docker and Docker Compose
- OpenAI API key
- Minimum 2GB RAM
- 5GB free disk space

### Environment Configuration

Create a `.env` file in the project root:

```bash
# Database Configuration
POSTGRES_PASSWORD=ragpassword
DATABASE_URL=postgresql://postgres:ragpassword@postgres:5432/ragdb

# Redis Configuration
REDIS_URL=redis://redis:6379

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Application Configuration
DEBUG=false
LOG_LEVEL=INFO
```

### Docker Deployment

1. Clone the repository:
```bash
git clone https://github.com/ABHAYMALLIK5566/Supernomics_task
cd Supernomics_task
```

2. Create the `.env` file with your OpenAI API key

3. Start the services:
```bash
docker-compose up -d --build
```

4. Verify deployment:
```bash
curl http://localhost:8001/health
```

## API Documentation

### Core Endpoints

#### Query Processing
```http
POST /query
Content-Type: application/json

{
    "query": "What are the fundamental principles of international law?",
    "top_k": 5,
    "use_agent": false,
    "algorithm": "hybrid",
    "similarity_threshold": 0.7,
    "text_only": false
}
```

#### Document Ingestion
```http
POST /ingest-pdfs
Content-Type: multipart/form-data

files: [PDF files]
```

#### Health Check
```http
GET /health
```

#### Service Information
```http
GET /
```

#### Adaptive Query Processing
```http
POST /adaptive-query
Content-Type: application/json

{
    "query": "Analyze the legal framework for human rights",
    "top_k": 5,
    "use_agent": false,
    "algorithm": "hybrid",
    "similarity_threshold": 0.7
}
```

#### Feedback Submission
```http
POST /feedback
Content-Type: application/json

{
    "query": "What are human rights?",
    "response": "Generated response text",
    "feedback_type": "rating",
    "rating": 4,
    "comments": "Good response"
}
```

### Response Format

```json
{
    "response": "Generated legal response text",
    "query": "Original user query",
    "context": [
        {
            "content": "Relevant document chunk",
            "metadata": {
                "title": "Document title",
                "source": "Document source",
                "similarity_score": 0.85
            }
        }
    ],
    "metadata": {
        "algorithm": "hybrid",
        "citations": ["Citation 1", "Citation 2"],
        "domain": "International Law"
    },
    "source": "rag_engine",
    "response_time_ms": 1500
}
```

## Configuration

### Application Settings

Key configuration parameters in `app/core/config.py`:

- `pdf_chunk_size`: Size of text chunks for processing (default: 1000)
- `pdf_chunk_overlap`: Overlap between chunks (default: 200)
- `max_text_length`: Maximum text length for processing (default: 8000)
- `similarity_threshold`: Minimum similarity score for retrieval (default: 0.7)
- `rag_top_k`: Number of top documents to retrieve (default: 5)
- `openai_embedding_model`: OpenAI embedding model (default: text-embedding-3-small)
- `openai_model`: OpenAI text generation model (default: gpt-4-turbo-preview)

### Database Configuration

PostgreSQL settings optimized for vector operations:
- `shared_buffers`: 256MB
- `effective_cache_size`: 1GB
- `maintenance_work_mem`: 64MB
- `random_page_cost`: 1.1

## Development Guidelines

### Code Structure

```
app/
├── api/                    # API endpoints and routing
├── core/                   # Core configuration and utilities
├── models/                 # Pydantic models for request/response
├── services/               # Business logic and service implementations
└── main.py                # FastAPI application entry point
```

### Error Handling

The system implements comprehensive error handling:

- **DocumentProcessingError**: For PDF processing failures
- **RateLimitExceededError**: For API rate limiting
- **QueryProcessingError**: For query processing failures
- **DatabaseError**: For database operation failures

### Logging

Structured logging is implemented throughout the system:

```python
import logging
logger = logging.getLogger(__name__)

# Log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
logger.info("Operation completed successfully")
logger.error("Operation failed", exc_info=True)
```

## Performance Optimization

### Database Optimization

1. **Indexing Strategy**:
   - Full-text search indexes on content
   - Vector similarity indexes on embeddings
   - Metadata indexes for filtering

2. **Query Optimization**:
   - Use prepared statements
   - Implement query result caching
   - Optimize vector similarity calculations

### Caching Strategy

1. **Redis Caching**:
   - Cache frequent query results
   - Store session data
   - Implement cache invalidation

2. **Application Caching**:
   - Cache document embeddings
   - Store processed metadata
   - Implement response caching

## Monitoring and Observability

### Health Checks

- **Application Health**: `/health` endpoint
- **Database Health**: Connection and query validation
- **Service Health**: Individual service status checks

### Metrics Collection

- Response time tracking
- Query success rates
- Document processing metrics
- Resource utilization monitoring

### Logging Strategy

- Structured JSON logging
- Error tracking and alerting
- Performance monitoring

## Security Considerations

### API Security

- Rate limiting per IP address
- Input validation and sanitization
- CORS configuration
- Environment variable validation

### Data Security

- Database connection encryption
- Secure environment variable handling
- Document access controls
- Audit logging for sensitive operations

## Command Reference

For detailed command usage and examples, see [COMMANDS.md](COMMANDS.md) which contains:

- System setup and configuration commands
- Docker management operations
- Database and Redis commands
- API testing and validation
- Document management operations
- System monitoring and troubleshooting
- Backup and recovery procedures
- Development and maintenance commands

## Troubleshooting

### Common Issues

1. **Database Connection Errors**:
   - Check PostgreSQL container status
   - Verify connection string
   - Check network connectivity

2. **OpenAI API Errors**:
   - Verify API key configuration
   - Check rate limits and quotas
   - Monitor API usage

3. **Memory Issues**:
   - Monitor container memory usage
   - Optimize document chunk sizes
   - Implement memory-efficient processing

### Debug Mode

Enable debug logging for troubleshooting:

```bash
export LOG_LEVEL=DEBUG
docker-compose up
```
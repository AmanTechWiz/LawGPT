# Legal Research Assistant - Command Reference

This document provides a comprehensive list of all commands and operations available in the Legal Research Assistant system.

## Table of Contents

1. [System Setup Commands](#system-setup-commands)
2. [Docker Management Commands](#docker-management-commands)
3. [Database Commands](#database-commands)
4. [Redis Commands](#redis-commands)
5. [API Testing Commands](#api-testing-commands)
6. [Document Management Commands](#document-management-commands)
7. [System Monitoring Commands](#system-monitoring-commands)
8. [Backup and Recovery Commands](#backup-and-recovery-commands)
9. [Troubleshooting Commands](#troubleshooting-commands)
10. [Development Commands](#development-commands)

## System Setup Commands

### Initial Setup
```bash
# Clone the repository
git clone https://github.com/ABHAYMALLIK5566/Supernomics_task.git

# Navigate to project directory
cd Supernomics_task

# Verify you're in the correct directory
pwd
ls -la

# Copy environment template
cp env.example .env

# Verify the .env file was created
ls -la .env

# Edit the .env file to add your OpenAI API key
nano .env
# OR use vim: vim .env
# OR use any text editor of your choice

# Add your OpenAI API key (replace with your actual key)
echo "OPENAI_API_KEY=your-actual-openai-api-key-here" >> .env

# Verify the API key was added
cat .env | grep OPENAI_API_KEY
```

### Environment Configuration
```bash
# Set production environment
export DEBUG=false
export LOG_LEVEL=INFO

# Update .env for production
echo "DEBUG=false" >> .env
echo "LOG_LEVEL=INFO" >> .env

# Check environment variables
cat .env
```


## Docker Management Commands

### Basic Docker Operations
```bash
# Start all services
docker-compose up -d

# Start all services with build
docker-compose up -d --build

# Stop all services
docker-compose down

# Stop all services and remove volumes
docker-compose down -v

# Restart all services
docker-compose restart

# Start specific service
docker-compose up -d rag-api
docker-compose up -d postgres
docker-compose up -d redis

# Stop specific service
docker-compose stop rag-api
docker-compose stop postgres
docker-compose stop redis
```

### Service Status and Monitoring
```bash
# Check service status
docker-compose ps

# Check service status with details
docker-compose ps --services

# View all logs
docker-compose logs

# View specific service logs
docker-compose logs rag-api
docker-compose logs postgres
docker-compose logs redis

# Follow logs in real-time
docker-compose logs -f rag-api
docker-compose logs -f postgres
docker-compose logs -f redis

# View last 100 lines of logs
docker-compose logs --tail=100 rag-api

# View logs with timestamps
docker-compose logs -t rag-api
```

### Container Operations
```bash
# Execute command in container
docker-compose exec rag-api bash
docker-compose exec postgres psql -U postgres ragdb
docker-compose exec redis redis-cli

# Run one-time command
docker-compose run rag-api bash

# Rebuild containers
docker-compose up -d --build

# Remove unused containers
docker-compose down --remove-orphans

# Pull latest images
docker-compose pull

# View container resource usage
docker stats

# View container details
docker inspect task_rag-api
```

### Docker System Management
```bash
# Clean up unused containers
docker container prune

# Clean up unused images
docker image prune

# Clean up unused volumes
docker volume prune

# Clean up everything
docker system prune -a

# View Docker system info
docker system df

# View Docker version
docker --version
docker-compose --version
```

## Database Commands

### PostgreSQL Connection and Management
```bash
# Connect to database
docker-compose exec postgres psql -U postgres ragdb

# Connect to database with specific host
docker-compose exec postgres psql -h localhost -U postgres ragdb

# Check database status
docker-compose exec postgres pg_isready -U postgres

# Check database version
docker-compose exec postgres psql -U postgres ragdb -c "SELECT version();"
```

### Database Operations
```bash
# List all tables
docker-compose exec postgres psql -U postgres ragdb -c "\dt"

# List all databases
docker-compose exec postgres psql -U postgres -c "\l"

# Check database size
docker-compose exec postgres psql -U postgres ragdb -c "SELECT pg_size_pretty(pg_database_size('ragdb'));"

# Count documents
docker-compose exec postgres psql -U postgres ragdb -c "SELECT COUNT(*) FROM documents;"

# Count documents with embeddings
docker-compose exec postgres psql -U postgres ragdb -c "SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL;"

# Get database statistics
docker-compose exec postgres psql -U postgres ragdb -c "SELECT * FROM get_database_stats();"

# Check table sizes
docker-compose exec postgres psql -U postgres ragdb -c "SELECT schemaname,tablename,pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size FROM pg_tables WHERE schemaname='public';"
```

### Database Maintenance
```bash
# Vacuum database
docker-compose exec postgres psql -U postgres ragdb -c "VACUUM ANALYZE;"

# Check database connections
docker-compose exec postgres psql -U postgres ragdb -c "SELECT count(*) FROM pg_stat_activity;"

# Check database configuration
docker-compose exec postgres psql -U postgres ragdb -c "SHOW ALL;"
```

### Database Backup and Restore
```bash
# Backup database
docker-compose exec postgres pg_dump -U postgres ragdb > backup_$(date +%Y%m%d).sql

# Backup database with compression
docker-compose exec postgres pg_dump -U postgres ragdb | gzip > backup_$(date +%Y%m%d).sql.gz

# Restore database
docker-compose exec -T postgres psql -U postgres ragdb < backup_20240101.sql

# Restore from compressed backup
gunzip -c backup_20240101.sql.gz | docker-compose exec -T postgres psql -U postgres ragdb

# Backup specific table
docker-compose exec postgres pg_dump -U postgres ragdb -t documents > documents_backup.sql

# Backup schema only
docker-compose exec postgres pg_dump -U postgres ragdb -s > schema_backup.sql

# Backup data only
docker-compose exec postgres pg_dump -U postgres ragdb -a > data_backup.sql
```

## Redis Commands

### Redis Connection and Management
```bash
# Connect to Redis
docker-compose exec redis redis-cli

# Check Redis status
docker-compose exec redis redis-cli ping

# Check Redis version
docker-compose exec redis redis-cli --version

# Check Redis info
docker-compose exec redis redis-cli INFO
```

### Redis Operations
```bash
# Check Redis memory usage
docker-compose exec redis redis-cli INFO memory

# Check Redis keys
docker-compose exec redis redis-cli KEYS "*"

# Count Redis keys
docker-compose exec redis redis-cli DBSIZE

# Clear all Redis data
docker-compose exec redis redis-cli FLUSHALL

# Clear specific database
docker-compose exec redis redis-cli FLUSHDB

# Monitor Redis operations
docker-compose exec redis redis-cli MONITOR

# Check Redis configuration
docker-compose exec redis redis-cli CONFIG GET "*"
```

### Redis Performance and Monitoring
```bash
# Check Redis latency
docker-compose exec redis redis-cli --latency

# Check Redis latency history
docker-compose exec redis redis-cli --latency-history

# Check Redis slow log
docker-compose exec redis redis-cli SLOWLOG GET 10

# Check Redis client list
docker-compose exec redis redis-cli CLIENT LIST

# Check Redis memory usage by key
docker-compose exec redis redis-cli --bigkeys
```

### Redis Backup and Persistence
```bash
# Save Redis data
docker-compose exec redis redis-cli SAVE

# Background save
docker-compose exec redis redis-cli BGSAVE

# Check last save time
docker-compose exec redis redis-cli LASTSAVE

# Copy Redis data file
docker-compose cp task_rag-redis-1:/data/dump.rdb ./redis_backup.rdb
```

## API Testing Commands

### Health and Status Checks
```bash
# Check API health
curl http://localhost:8001/health

# Check API health with verbose output
curl -v http://localhost:8001/health

# Check API health with timing
time curl http://localhost:8001/health

# Check service info
curl http://localhost:8001/

# Check API documentation
curl http://localhost:8001/docs

# Check OpenAPI schema
curl http://localhost:8001/openapi.json
```

### Query Processing Tests
```bash
# Simple query test
curl -X POST "http://localhost:8001/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Article 1 of the UN Charter?", "top_k": 2}'

# Complex query test
curl -X POST "http://localhost:8001/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "List all enforcement measures in the UN Charter", "top_k": 3}'

# Multi-hop reasoning test
curl -X POST "http://localhost:8001/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Compare Article 41 and Article 42 of the UN Charter", "enable_multi_hop_reasoning": true, "session_id": "test_session"}'

# JSON query test
curl -X POST "http://localhost:8001/query-json" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are fundamental rights?", "top_k": 5}'

# Text-only query test
curl -X POST "http://localhost:8001/query-text" \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain the UN Charter", "top_k": 3}'
```

### Adaptive Query Tests
```bash
# Test adaptive query endpoint
curl -X POST "http://localhost:8001/adaptive-query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Article 1 of the UN Charter?", "top_k": 2}'

# Test with different similarity thresholds
curl -X POST "http://localhost:8001/adaptive-query" \
  -H "Content-Type: application/json" \
  -d '{"query": "UN Charter enforcement", "top_k": 5, "similarity_threshold": 0.3}'

# Test with different algorithms
curl -X POST "http://localhost:8001/adaptive-query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Legal analysis", "top_k": 3, "algorithm": "hybrid"}'
```

### Streaming Tests
```bash
# Test streaming endpoint
curl -X POST "http://localhost:8001/stream" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the UN Charter?", "top_k": 2}'

# Test streaming with agent
curl -X POST "http://localhost:8001/stream" \
  -H "Content-Type: application/json" \
  -d '{"query": "Analyze the UN Charter", "use_agent": true, "top_k": 3}'
```

### Rate Limiting Tests
```bash
# Test rate limiting
for i in {1..10}; do
  curl -X POST "http://localhost:8001/query" \
    -H "Content-Type: application/json" \
    -d '{"query": "Test query '${i}'", "top_k": 1}'
  echo "Request $i completed"
done
```

## Document Management Commands

### Document Ingestion
```bash
# Ingest PDF files via API
curl -X POST "http://localhost:8001/ingest-pdfs" \
  -F "files=@sample_documents/uncharter.pdf"

# Ingest multiple PDF files
curl -X POST "http://localhost:8001/ingest-pdfs" \
  -F "files=@sample_documents/uncharter.pdf" \
  -F "files=@sample_documents/crc.pdf"
```

### Document Operations
```bash
# Get document statistics
docker-compose exec postgres psql -U postgres ragdb -c "SELECT source, COUNT(*) as count FROM documents GROUP BY source;"

# Get document content length statistics
docker-compose exec postgres psql -U postgres ragdb -c "SELECT MIN(LENGTH(content)) as min_length, MAX(LENGTH(content)) as max_length, AVG(LENGTH(content)) as avg_length FROM documents;"
```

### Document Search and Analysis
```bash
# Search documents by content
docker-compose exec postgres psql -U postgres ragdb -c "SELECT id, title, source FROM documents WHERE content ILIKE '%UN Charter%' LIMIT 5;"

# Search documents by title
docker-compose exec postgres psql -U postgres ragdb -c "SELECT id, title, source FROM documents WHERE title ILIKE '%constitution%' LIMIT 5;"

# Get documents without embeddings
docker-compose exec postgres psql -U postgres ragdb -c "SELECT id, title, source FROM documents WHERE embedding IS NULL LIMIT 5;"

# Get embedding statistics
docker-compose exec postgres psql -U postgres ragdb -c "SELECT COUNT(*) as total_docs, COUNT(embedding) as docs_with_embeddings, ROUND(COUNT(embedding)::numeric / COUNT(*) * 100, 2) as embedding_coverage FROM documents;"
```

## System Monitoring Commands

### Resource Monitoring
```bash
# Check container resource usage
docker stats

# Check system resources
htop
free -h
df -h

# Check disk usage
du -sh .
du -sh sample_documents/

# Check network connections
ss -tulpn | grep :8001
ss -tulpn | grep :5432
ss -tulpn | grep :6379

# Alternative using lsof
lsof -i :8001
lsof -i :5432
lsof -i :6379

# Check system load
uptime
top
```

### Log Management
```bash
# Save logs to file
docker-compose logs rag-api > api_logs.txt
docker-compose logs postgres > postgres_logs.txt
docker-compose logs redis > redis_logs.txt

# Search logs for errors
docker-compose logs rag-api | grep ERROR
docker-compose logs postgres | grep ERROR
docker-compose logs redis | grep ERROR

# Search logs for specific text
docker-compose logs rag-api | grep "query"
docker-compose logs rag-api | grep "rate limit"
docker-compose logs rag-api | grep "database"
```

### Performance Monitoring
```bash
# Check API response time
time curl http://localhost:8001/health

# Test with verbose output
curl -v http://localhost:8001/health

# Check database performance
docker-compose exec postgres psql -U postgres ragdb -c "EXPLAIN ANALYZE SELECT * FROM documents LIMIT 10;"

# Check Redis performance
docker-compose exec redis redis-cli --latency

# Check database connections
docker-compose exec postgres psql -U postgres ragdb -c "SELECT count(*) FROM pg_stat_activity;"
```

## Backup and Recovery Commands

### Complete System Backup
```bash
# Create backup directory
mkdir -p backups/$(date +%Y%m%d)

# Backup database
docker-compose exec postgres pg_dump -U postgres ragdb > backups/$(date +%Y%m%d)/database_backup.sql

# Backup Redis data
docker-compose exec redis redis-cli BGSAVE
docker-compose cp task_rag-redis-1:/data/dump.rdb backups/$(date +%Y%m%d)/redis_backup.rdb

# Backup application files
tar -czf backups/$(date +%Y%m%d)/app_backup.tar.gz app/

# Backup configuration files
cp docker-compose.yml backups/$(date +%Y%m%d)/
cp .env backups/$(date +%Y%m%d)/
cp requirements.txt backups/$(date +%Y%m%d)/

# Backup sample documents
tar -czf backups/$(date +%Y%m%d)/sample_documents_backup.tar.gz sample_documents/

# Create complete backup archive
tar -czf complete_backup_$(date +%Y%m%d).tar.gz backups/$(date +%Y%m%d)/
```

### Database Backup and Restore
```bash
# Backup with timestamp
docker-compose exec postgres pg_dump -U postgres ragdb > backup_$(date +%Y%m%d_%H%M%S).sql

# Backup with compression
docker-compose exec postgres pg_dump -U postgres ragdb | gzip > backup_$(date +%Y%m%d_%H%M%S).sql.gz

# Restore from backup
docker-compose exec -T postgres psql -U postgres ragdb < backup_20240101_120000.sql

# Restore from compressed backup
gunzip -c backup_20240101_120000.sql.gz | docker-compose exec -T postgres psql -U postgres ragdb

# Verify backup integrity
docker-compose exec postgres pg_dump -U postgres ragdb --schema-only > schema_check.sql
```

### Volume Backup
```bash
# Backup PostgreSQL volume
docker run --rm -v supernomics_task_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup_$(date +%Y%m%d).tar.gz -C /data .

# Backup Redis volume
docker run --rm -v supernomics_task_redis_data:/data -v $(pwd):/backup alpine tar czf /backup/redis_backup_$(date +%Y%m%d).tar.gz -C /data .

# Restore PostgreSQL volume
docker run --rm -v supernomics_task_postgres_data:/data -v $(pwd):/backup alpine tar xzf /backup/postgres_backup_20240101.tar.gz -C /data

# Restore Redis volume
docker run --rm -v supernomics_task_redis_data:/data -v $(pwd):/backup alpine tar xzf /backup/redis_backup_20240101.tar.gz -C /data
```

## Troubleshooting Commands

### Common Issues
```bash

# Kill process using port
kill -9 $(lsof -t -i:8001)
kill -9 $(lsof -t -i:5432)
kill -9 $(lsof -t -i:6379)

# Check Docker daemon
sudo systemctl status docker
sudo systemctl restart docker

# Check Docker logs
sudo journalctl -u docker.service

# Check container logs
docker logs task_rag-api
docker logs task_rag-postgres-1
docker logs task_rag-redis-1
```

### Service Debugging
```bash
# Check service health
curl http://localhost:8001/health

# Check database connection
docker-compose exec postgres pg_isready -U postgres

# Check Redis connection
docker-compose exec redis redis-cli ping

# Check container status
docker-compose ps

# Check container resource usage
docker stats

# Check container logs
docker-compose logs --tail=50 rag-api
docker-compose logs --tail=50 postgres
docker-compose logs --tail=50 redis
```

### Performance Debugging
```bash
# Test API response time
time curl http://localhost:8001/health

# Test with verbose output
curl -v http://localhost:8001/health

# Check database performance
docker-compose exec postgres psql -U postgres ragdb -c "EXPLAIN ANALYZE SELECT * FROM documents LIMIT 10;"

# Check Redis performance
docker-compose exec redis redis-cli --latency

# Check system resources
free -h
df -h
htop
```

### Error Investigation
```bash
# Search for errors in logs
docker-compose logs rag-api | grep -i error
docker-compose logs postgres | grep -i error
docker-compose logs redis | grep -i error

# Search for warnings
docker-compose logs rag-api | grep -i warning
docker-compose logs postgres | grep -i warning
docker-compose logs redis | grep -i warning

# Check recent logs
docker-compose logs --since=1h rag-api
docker-compose logs --since=1h postgres
docker-compose logs --since=1h redis
```

## Development Commands

### Code Development
```bash
# Run application locally (without Docker)
cd app
python -m uvicorn main:app --host 0.0.0.0 --port 8001 --reload

# Install dependencies locally
pip install -r requirements.txt

# Run tests (if available)
python -m pytest tests/

# Run linting (if available)
python -m flake8 app/
python -m black app/

# Format code
python -m black app/
python -m isort app/
```

### Database Development
```bash
# Connect to database for development
docker-compose exec postgres psql -U postgres ragdb

# Create database backup for development
docker-compose exec postgres pg_dump -U postgres ragdb > dev_backup.sql

# Reset database
docker-compose exec postgres psql -U postgres ragdb -c "DROP SCHEMA public CASCADE; CREATE SCHEMA public;"
docker-compose exec postgres psql -U postgres ragdb -f /docker-entrypoint-initdb.d/init.sql
```

### Testing and Validation
```bash
# Test all endpoints
curl http://localhost:8001/health
curl http://localhost:8001/
curl http://localhost:8001/docs

# Test query processing
curl -X POST "http://localhost:8001/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "Test query", "top_k": 1}'

# Test document ingestion
curl -X POST "http://localhost:8001/ingest-pdfs" \
  -F "files=@sample_documents/uncharter.pdf"
```

### Maintenance Commands
```bash
# Update system
docker-compose pull
docker-compose up -d --build

# Clean up system
docker system prune -a
docker volume prune

# Restart services
docker-compose restart

# Check system status
docker-compose ps
curl http://localhost:8001/health
```

## Feedback System Commands
```bash
# Submit feedback
curl -X POST "http://localhost:8001/feedback" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are human rights?", "response": "Generated response", "feedback_type": "rating", "rating": 4, "comments": "Good response"}'

# Get feedback metrics
curl http://localhost:8001/feedback/metrics

# Get feedback analysis
curl http://localhost:8001/feedback/analysis

# Get recent feedback
curl http://localhost:8001/feedback/recent
```

## Production Deployment Commands

### Production Setup
```bash
# Set production environment
export DEBUG=false
export LOG_LEVEL=INFO

# Update .env for production
echo "DEBUG=false" >> .env
echo "LOG_LEVEL=INFO" >> .env

# Start with production settings
docker-compose up -d

# Monitor production logs
docker-compose logs -f rag-api
```

### Production Monitoring
```bash
# Monitor system health
curl http://localhost:8001/health

# Monitor resource usage
docker stats

# Monitor logs
docker-compose logs --tail=100 -f rag-api

# Check database performance
docker-compose exec postgres psql -U postgres ragdb -c "SELECT * FROM get_database_stats();"
```

### Production Maintenance
```bash
# Backup production data
docker-compose exec postgres pg_dump -U postgres ragdb > production_backup_$(date +%Y%m%d).sql

# Update production system
docker-compose pull
docker-compose up -d --build

# Monitor update
docker-compose logs -f rag-api
```
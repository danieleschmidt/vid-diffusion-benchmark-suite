# Operational Procedures for Video Diffusion Benchmark Suite

## Overview

This document outlines standard operational procedures for maintaining and operating the Video Diffusion Benchmark Suite in production environments.

## Daily Operations

### Health Checks

#### System Health
```bash
# Check service status
docker-compose ps

# Check resource usage
docker stats

# Check GPU status
nvidia-smi

# Check disk space
df -h

# Check memory usage
free -h
```

#### Application Health
```bash
# Check API health endpoint
curl -f http://localhost:8000/health

# Check dashboard accessibility
curl -f http://localhost:8501

# Check metrics endpoint
curl -f http://localhost:9090/metrics

# Validate model availability
vid-bench list-models
```

#### Performance Metrics
```bash
# Check recent benchmark performance
vid-bench status --last-24h

# Monitor queue status
redis-cli llen benchmark_queue

# Check error rates
grep ERROR /var/log/vid-bench/*.log | wc -l

# Monitor resource utilization
iostat 1 3
```

### Log Monitoring

#### Critical Log Patterns
Monitor for these patterns in logs:
- `ERROR` - Application errors
- `CUDA out of memory` - GPU memory issues
- `Connection refused` - Service connectivity issues
- `Timeout` - Performance degradation
- `Failed to load model` - Model loading issues

#### Log Commands
```bash
# View recent errors
tail -f /var/log/vid-bench/error.log

# Search for specific patterns
grep -i "cuda out of memory" /var/log/vid-bench/*.log

# Count error occurrences
grep -c ERROR /var/log/vid-bench/app.log

# Monitor real-time logs
docker-compose logs -f benchmark
```

## Weekly Operations

### Performance Review

#### Benchmark Metrics Analysis
```bash
# Generate weekly performance report
vid-bench report --period=week --output=/reports/weekly.html

# Analyze model performance trends
vid-bench analyze trends --models=all --period=week

# Check for performance regressions
vid-bench regression-test --baseline=last-week
```

#### Resource Usage Review
- Review CPU and memory usage patterns
- Analyze GPU utilization rates
- Check storage growth and cleanup needs
- Monitor network traffic patterns

### Maintenance Tasks

#### System Updates
```bash
# Update system packages
sudo apt update && sudo apt upgrade

# Update Docker images
docker-compose pull
docker system prune -f

# Update Python dependencies
pip-review --local --auto

# Clear temporary files
find /tmp -name "vid-bench-*" -mtime +7 -delete
```

#### Database Maintenance
```bash
# Backup results database
pg_dump vid_bench > /backup/vid_bench_$(date +%Y%m%d).sql

# Vacuum database
psql -d vid_bench -c "VACUUM ANALYZE;"

# Clean old results (older than 90 days)
vid-bench cleanup --older-than=90d
```

## Monthly Operations

### Capacity Planning

#### Storage Analysis
```bash
# Analyze storage usage by category
du -sh /data/models/* | sort -hr
du -sh /data/results/* | sort -hr
du -sh /data/cache/* | sort -hr

# Check for storage growth trends
df -h | grep -E "(models|results|cache)"

# Clean up old cache files
find /data/cache -type f -mtime +30 -delete
```

#### Performance Trending
- Analyze monthly performance reports
- Identify capacity bottlenecks
- Plan hardware upgrades
- Review SLA compliance

### Model Management

#### Model Updates
```bash
# Check for new model versions
vid-bench check-updates

# Update model registry
vid-bench update-registry

# Test new models
vid-bench test-models --new-only

# Deploy approved models
vid-bench deploy-models --staging-first
```

#### Model Cleanup
```bash
# Remove unused model versions
vid-bench cleanup-models --keep-latest=3

# Archive old benchmark results
vid-bench archive --older-than=6m

# Update model metadata
vid-bench refresh-metadata --all
```

## Incident Response

### Severity Levels

#### Critical (P0)
- Service completely down
- Data corruption detected
- Security breach
- GPU hardware failure

**Response Time**: 15 minutes
**Resolution Time**: 4 hours

#### High (P1)
- Significant performance degradation
- Model failures affecting multiple users
- Authentication issues
- Monitoring system failure

**Response Time**: 1 hour
**Resolution Time**: 24 hours

#### Medium (P2)
- Single model failures
- Dashboard issues
- Non-critical feature failures
- Performance warnings

**Response Time**: 4 hours
**Resolution Time**: 3 days

#### Low (P3)
- Minor UI issues
- Documentation problems
- Enhancement requests
- Cosmetic issues

**Response Time**: 1 day
**Resolution Time**: 1 week

### Common Issues and Solutions

#### GPU Out of Memory
```bash
# Immediate mitigation
docker-compose restart benchmark

# Check memory usage
nvidia-smi

# Reduce batch sizes
vid-bench config set batch_size 1

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

#### Service Unresponsive
```bash
# Check process status
ps aux | grep vid-bench

# Check port availability
netstat -tlnp | grep :8000

# Restart services
docker-compose restart

# Check logs for root cause
docker-compose logs --tail=100 benchmark
```

#### Database Connection Issues
```bash
# Check database status
pg_isready -h localhost -p 5432

# Test connection
psql -h localhost -U vid_bench -d vid_bench -c "SELECT 1;"

# Restart database
docker-compose restart postgres

# Check connection pool
vid-bench db status
```

#### Model Loading Failures
```bash
# Check model file integrity
vid-bench verify-models

# Re-download corrupted models
vid-bench download-model --force MODEL_NAME

# Check disk space
df -h /data/models

# Clear model cache
rm -rf /data/cache/models/*
```

## Disaster Recovery

### Backup Procedures

#### Daily Backups
```bash
#!/bin/bash
# /scripts/daily_backup.sh

DATE=$(date +%Y%m%d)
BACKUP_DIR="/backup/$DATE"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup database
pg_dump vid_bench > $BACKUP_DIR/database.sql

# Backup configuration
cp -r /config $BACKUP_DIR/

# Backup critical results
rsync -av /data/results/critical/ $BACKUP_DIR/results/

# Upload to cloud storage
aws s3 sync $BACKUP_DIR s3://vid-bench-backups/$DATE/
```

#### Weekly Full Backups
```bash
#!/bin/bash
# /scripts/weekly_backup.sh

DATE=$(date +%Y%m%d)
BACKUP_DIR="/backup/full-$DATE"

# Full system backup
tar -czf $BACKUP_DIR/system.tar.gz \
  --exclude='/data/cache' \
  --exclude='/tmp' \
  /data /config /scripts

# Upload to long-term storage
aws s3 cp $BACKUP_DIR/system.tar.gz s3://vid-bench-backups/weekly/
```

### Recovery Procedures

#### Database Recovery
```bash
# Stop services
docker-compose stop

# Restore database
psql -d vid_bench < /backup/latest/database.sql

# Verify integrity
vid-bench db verify

# Restart services
docker-compose start
```

#### Full System Recovery
```bash
# Restore system files
tar -xzf /backup/latest/system.tar.gz -C /

# Restore configuration
cp -r /backup/latest/config/* /config/

# Restart all services
docker-compose up -d

# Verify system health
vid-bench health-check --comprehensive
```

## Monitoring and Alerting

### Alert Definitions

#### Critical Alerts
- Service down > 5 minutes
- GPU temperature > 85°C
- Disk usage > 90%
- Memory usage > 95%
- Error rate > 10%

#### Warning Alerts
- Response time > 30 seconds
- GPU utilization < 10% for 1 hour
- Queue length > 100
- Failed benchmarks > 5%
- Disk usage > 80%

### Alert Response

#### Immediate Actions
1. Acknowledge alert
2. Check service status
3. Review recent logs
4. Implement immediate mitigation
5. Escalate if necessary

#### Follow-up Actions
1. Root cause analysis
2. Implement permanent fix
3. Update documentation
4. Review alert thresholds
5. Post-incident review

## Performance Optimization

### Regular Optimization Tasks

#### GPU Optimization
```bash
# Check GPU utilization patterns
nvidia-smi dmon -s m

# Optimize CUDA settings
export CUDA_LAUNCH_BLOCKING=1
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024

# Clear GPU memory periodically
python -c "import torch; torch.cuda.empty_cache()"
```

#### Memory Optimization
```bash
# Monitor memory usage
top -p $(pgrep -f vid-bench)

# Clear system caches
echo 3 > /proc/sys/vm/drop_caches

# Optimize Python memory
export PYTHONMALLOC=malloc
export MALLOC_TRIM_THRESHOLD_=100000
```

#### Storage Optimization
```bash
# Clean temporary files
find /tmp -name "*.tmp" -mtime +1 -delete

# Compress old logs
gzip /var/log/vid-bench/*.log.1

# Optimize database
VACUUM ANALYZE;
REINDEX DATABASE vid_bench;
```

## Security Operations

### Daily Security Checks
```bash
# Check for unauthorized access
grep "Failed password" /var/log/auth.log

# Monitor unusual network activity
netstat -tuln | grep LISTEN

# Check file integrity
aide --check

# Scan for malware
clamscan -r /data --log=/var/log/clamav.log
```

### Weekly Security Tasks
- Update security patches
- Review access logs
- Audit user permissions
- Check SSL certificate expiry
- Scan for vulnerabilities

### Monthly Security Reviews
- Security policy review
- Incident analysis
- Penetration testing
- Backup encryption verification
- Access control audit

## Documentation Updates

### Procedure Updates
- Review procedures quarterly
- Update based on incidents
- Incorporate team feedback
- Version control changes
- Train team on updates

### Knowledge Base Maintenance
- Update troubleshooting guides
- Document new procedures
- Archive obsolete information
- Maintain FAQ sections
- Create training materials

This operational runbook provides comprehensive procedures for maintaining the Video Diffusion Benchmark Suite in production environments, ensuring reliable operation and quick incident resolution.
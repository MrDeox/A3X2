# 📊 Monitoring and Observability

This guide covers monitoring A3X deployments, tracking performance, and maintaining system health.

## Why Monitor A3X?

Effective monitoring helps you:

- **Track performance** and identify bottlenecks
- **Ensure reliability** and catch issues early
- **Optimize costs** by identifying inefficiencies
- **Maintain security** through audit trails
- **Plan capacity** based on usage patterns

## Key Metrics to Monitor

### 1. A3X Service Health

**Essential metrics:**

| Metric | Description | Threshold | Action |
|--------|-------------|-----------|--------|
| **Service Status** | Is A3X running? | Down > 5min | Restart service |
| **Response Time** | Time to process requests | > 30s avg | Investigate |
| **Error Rate** | Percentage of failed operations | > 5% | Review logs |
| **Memory Usage** | RAM consumption | > 80% | Scale up |
| **CPU Usage** | Processor utilization | > 70% | Investigate |

### 2. API Usage and Costs

**Cost-related metrics:**

| Metric | Description | Threshold | Action |
|--------|-------------|-----------|--------|
| **API Calls/min** | Rate of AI API usage | Varies | Monitor budget |
| **Token Usage** | AI model token consumption | Budget limit | Alert |
| **Cost per Hour** | Running cost rate | $10/hour | Review |
| **Success Rate** | Successful operations % | < 90% | Investigate |

### 3. Application Health

**Application-specific metrics:**

| Metric | Description | Threshold | Action |
|--------|-------------|-----------|--------|
| **Test Success Rate** | Passing tests % | < 95% | Fix tests |
| **Code Coverage** | Test coverage % | < 80% | Add tests |
| **Build Status** | CI/CD pipeline health | Any failure | Fix build |
| **Performance** | Response times | > 2s | Optimize |

## Monitoring Tools and Setup

### 1. System Monitoring

#### Using Prometheus + Grafana (Recommended)

1. **Set up Prometheus**:
   ```bash
   a3x run --goal "Configure Prometheus metrics collection for A3X" --config configs/sample.yaml
   ```

2. **Create Grafana dashboards**:
   ```bash
   a3x run --goal "Create Grafana dashboards for A3X monitoring" --config configs/sample.yaml
   ```

3. **Configure alerting**:
   ```bash
   a3x run --goal "Set up alerting rules for critical A3X metrics" --config configs/sample.yaml
   ```

#### Simple System Monitoring

**Basic health checks**:
```bash
#!/bin/bash
# Simple monitoring script

# Check if A3X service is running
if ! systemctl is-active --quiet a3x; then
    echo "$(date): A3X service is down!" >> /var/log/a3x-monitor.log
    systemctl restart a3x
fi

# Check disk space
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 80 ]; then
    echo "$(date): High disk usage: ${DISK_USAGE}%" >> /var/log/a3x-monitor.log
fi

# Check memory usage
MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.0f", $3/$2 * 100.0}')
if [ "$MEMORY_USAGE" -gt 85 ]; then
    echo "$(date): High memory usage: ${MEMORY_USAGE}%" >> /var/log/a3x-monitor.log
fi
```

### 2. Log Monitoring

#### Centralized Logging

1. **Set up log aggregation**:
   ```bash
   a3x run --goal "Configure centralized logging with Elasticsearch" --config configs/sample.yaml
   ```

2. **Create log parsing**:
   ```bash
   a3x run --goal "Create log parsing rules for A3X logs" --config configs/sample.yaml
   ```

#### Log Analysis

**Key log patterns to monitor:**

```bash
# Error patterns
grep "ERROR" /var/log/a3x.log | tail -10
grep "Exception" /var/log/a3x.log | tail -10

# Performance issues
grep "timeout" /var/log/a3x.log | tail -10
grep "slow" /var/log/a3x.log | tail -10

# API issues
grep "rate limit" /var/log/a3x.log | tail -10
grep "authentication" /var/log/a3x.log | tail -10
```

### 3. Application Monitoring

#### Custom Health Checks

1. **Create health check endpoint**:
   ```bash
   a3x run --goal "Add health check endpoint to the application" --config configs/sample.yaml
   ```

2. **Set up monitoring**:
   ```bash
   # Monitor application health
   curl -f http://localhost:8000/health || echo "App health check failed"

   # Monitor database connectivity
   a3x run --goal "Create database connectivity monitoring" --config configs/sample.yaml
   ```

## Alerting and Notifications

### Setting Up Alerts

#### Email Alerts

```bash
a3x run --goal "Configure email alerts for critical A3X issues" --config configs/sample.yaml
```

**Alert conditions:**
- Service downtime > 5 minutes
- Error rate > 10%
- API costs > budget threshold
- Disk space > 90%

#### Slack/Discord Notifications

```bash
a3x run --goal "Set up Slack notifications for A3X events" --config configs/sample.yaml
```

### Alert Management

#### Escalation Procedures

1. **Level 1**: Automatic retry/restart
2. **Level 2**: Email/Slack notification
3. **Level 3**: Phone/SMS alert
4. **Level 4**: Manual intervention

#### Alert Fatigue Prevention

- **Set appropriate thresholds**
- **Use alerting delays** for transient issues
- **Group related alerts**
- **Regular alert review and tuning**

## Performance Monitoring

### 1. Resource Monitoring

#### Memory Monitoring

```bash
# Monitor memory usage trends
a3x run --goal "Create memory usage monitoring and alerting" --config configs/sample.yaml
```

**Memory optimization tips:**
- Monitor for memory leaks
- Set appropriate JVM heap sizes
- Clean up temporary files regularly
- Use memory-efficient algorithms

#### CPU Monitoring

```bash
# Monitor CPU usage patterns
a3x run --goal "Set up CPU usage monitoring and optimization" --config configs/sample.yaml
```

### 2. API Performance Monitoring

#### Response Time Tracking

```bash
a3x run --goal "Implement API response time monitoring" --config configs/sample.yaml
```

**Performance benchmarks:**
- **Simple operations**: < 5 seconds
- **Medium tasks**: < 30 seconds
- **Complex projects**: < 5 minutes

#### Throughput Monitoring

```bash
a3x run --goal "Monitor A3X operation throughput and capacity" --config configs/sample.yaml
```

## Cost Monitoring

### API Cost Tracking

#### Real-time Cost Monitoring

1. **Set up cost tracking**:
   ```bash
   a3x run --goal "Implement real-time API cost monitoring" --config configs/sample.yaml
   ```

2. **Create cost dashboards**:
   ```bash
   a3x run --goal "Build cost monitoring dashboard" --config configs/sample.yaml
   ```

#### Budget Alerts

```yaml
# Cost monitoring configuration
monitoring:
  cost_tracking: true
  budget_limit: 50.00  # USD per day
  alert_threshold: 80   # Alert at 80% of budget
  cost_reports: true   # Generate daily reports
```

### Cost Optimization

#### Identifying Cost Drivers

1. **Analyze usage patterns**:
   ```bash
   a3x run --goal "Analyze API usage patterns and identify cost optimization opportunities" --config configs/sample.yaml
   ```

2. **Optimize model selection**:
   ```bash
   a3x run --goal "Optimize AI model selection for cost efficiency" --config configs/sample.yaml
   ```

#### Cost Reduction Strategies

- **Use efficient models** for simple tasks
- **Implement caching** for repeated operations
- **Batch similar operations** together
- **Set usage quotas** and limits

## Security Monitoring

### Access Monitoring

#### API Key Monitoring

```bash
a3x run --goal "Monitor API key usage and security" --config configs/sample.yaml
```

**Monitor for:**
- Unusual usage patterns
- Geographic anomalies
- Rate limit violations
- Failed authentication attempts

#### File System Monitoring

```bash
a3x run --goal "Set up file system access monitoring" --config configs/sample.yaml
```

**Track:**
- File modifications outside workspace
- Unauthorized file access
- Large file operations
- Sensitive file changes

### Audit Logging

#### Comprehensive Audit Trail

```yaml
# Audit logging configuration
audit:
  enabled: true
  log_file: "/var/log/a3x-audit.log"
  log_level: "INFO"
  include:
    - "code_changes"
    - "file_operations"
    - "api_calls"
    - "authentication"
```

## Maintenance and Health Checks

### Regular Health Checks

#### Daily Health Checks

```bash
#!/bin/bash
# Daily health check script

echo "=== A3X Daily Health Check ==="

# Check service status
systemctl status a3x --no-pager

# Check recent errors
echo "Recent errors:"
tail -20 /var/log/a3x.log | grep -i error || echo "No recent errors"

# Check disk usage
echo "Disk usage:"
df -h /var/log/a3x.log | tail -1

# Check API connectivity
echo "API connectivity test:"
curl -s -H "Authorization: Bearer $OPENROUTER_API_KEY" \
     https://openrouter.ai/api/v1/models | head -1 || echo "API test failed"

echo "=== Health Check Complete ==="
```

#### Weekly Maintenance

```bash
a3x run --goal "Perform weekly maintenance and optimization" --config configs/sample.yaml
```

**Weekly tasks:**
- Clean up old log files
- Update system packages
- Review and rotate API keys
- Check and optimize performance

### Automated Monitoring Setup

#### One-Click Monitoring Setup

```bash
a3x run --goal "Set up comprehensive monitoring stack with Prometheus, Grafana, and alerting" --config configs/sample.yaml
```

**Includes:**
- Automated metric collection
- Pre-configured dashboards
- Alert rule templates
- Documentation and runbooks

## Troubleshooting Monitoring Issues

### Common Monitoring Problems

#### Metrics Not Collecting

**Problem:** Monitoring tools not receiving metrics.

**Solutions:**
1. **Check service status**:
   ```bash
   systemctl status a3x
   ```

2. **Verify configuration**:
   ```bash
   a3x run --goal "Validate monitoring configuration" --config configs/sample.yaml
   ```

3. **Test metric endpoints**:
   ```bash
   curl http://localhost:9090/metrics
   ```

#### False Alerts

**Problem:** Getting alerts for normal behavior.

**Solutions:**
1. **Adjust alert thresholds**:
   ```yaml
   alerting:
     thresholds:
       error_rate: 0.05    # 5% instead of 1%
       response_time: 60   # 60s instead of 30s
   ```

2. **Add alert delays**:
   ```yaml
   alerting:
     delays:
       minor_issues: 300   # 5 minute delay
       major_issues: 60    # 1 minute delay
   ```

#### Performance Impact

**Problem:** Monitoring itself affects performance.

**Solutions:**
1. **Optimize collection frequency**:
   ```yaml
   monitoring:
     collection_interval: 60  # Every 60s instead of 15s
   ```

2. **Sample metrics**:
   ```yaml
   monitoring:
     sampling_rate: 0.1  # Sample 10% of requests
   ```

## Monitoring Best Practices

### 1. Start Simple

**Begin with essentials:**
- Service up/down status
- Basic error monitoring
- Disk space alerts
- API cost tracking

**Expand gradually:**
- Detailed performance metrics
- Business-specific KPIs
- Predictive monitoring
- Automated responses

### 2. Set Meaningful Thresholds

**Avoid alert fatigue:**
- Set thresholds based on normal behavior
- Use statistical baselines
- Consider business impact
- Review and adjust regularly

### 3. Document Everything

**Maintain monitoring documentation:**
- Alert meanings and responses
- Troubleshooting procedures
- Configuration changes
- Performance benchmarks

### 4. Regular Review

**Schedule regular reviews:**
- Weekly: Alert review and tuning
- Monthly: Performance analysis
- Quarterly: Architecture review
- Annually: Tool evaluation

## Getting Help

### Monitoring Support

**When monitoring issues arise:**
1. **Check service logs** for error patterns
2. **Verify configuration** against documentation
3. **Test connectivity** to monitoring services
4. **Review recent changes** that might affect monitoring

**Resources:**
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [A3X Troubleshooting Guide](../troubleshooting/common-issues.md)

---

**Effective monitoring** ensures your A3X deployment runs smoothly and helps you catch issues before they become problems. Start with the basics and expand as your needs grow.
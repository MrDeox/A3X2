# 🚀 Deployment Guide

This guide covers deploying A3X-powered applications and services to production environments.

## Deployment Strategies

### 1. Application Deployment

When deploying applications built or modified by A3X, consider these approaches:

#### Container Deployment (Recommended)

**Docker Deployment:**

1. **Create Dockerfile** (ask A3X to help):
   ```bash
   a3x run --goal "Create Docker configuration for the application" --config configs/sample.yaml
   ```

2. **Build and test container**:
   ```bash
   docker build -t my-app:latest .
   docker run -p 8000:8000 my-app:latest
   ```

3. **Deploy to container platform**:
   ```bash
   # Push to registry
   docker tag my-app:latest my-registry/my-app:latest
   docker push my-registry/my-app:latest

   # Deploy to Kubernetes
   kubectl apply -f k8s-deployment.yaml
   ```

#### Traditional Server Deployment

**Linux Server Deployment:**

1. **Set up production server**:
   ```bash
   sudo apt-get update
   sudo apt-get install python3.10 python3.10-venv nginx
   ```

2. **Deploy application code**:
   ```bash
   # Copy files to server
   scp -r /local/project user@server:/opt/my-app/

   # Set up Python environment
   ssh user@server "cd /opt/my-app && python3.10 -m venv venv"
   ssh user@server "cd /opt/my-app && ./venv/bin/pip install -r requirements.txt"
   ```

3. **Configure web server** (nginx example):
   ```bash
   a3x run --goal "Create nginx configuration for the application" --config configs/sample.yaml
   ```

### 2. A3X Service Deployment

For running A3X as a service in production:

#### Systemd Service (Linux)

1. **Create service file**:
   ```bash
   a3x run --goal "Create systemd service configuration for A3X daemon" --config configs/sample.yaml
   ```

2. **Install and enable service**:
   ```bash
   sudo cp a3x.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable a3x
   sudo systemctl start a3x
   ```

#### Docker Container

1. **Create A3X container**:
   ```bash
   a3x run --goal "Create Docker container for running A3X as a service" --config configs/sample.yaml
   ```

2. **Run containerized A3X**:
   ```bash
   docker run -d --name a3x-service \
     -v /host/projects:/projects \
     -e OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
     a3x-service:latest
   ```

## Environment Configuration

### Production Configuration

**Key differences from development:**

```yaml
# Production config
llm:
  model: "x-ai/grok-beta"  # More stable model
  max_tokens: 2000        # Reduce for cost efficiency

execution:
  command_timeout: 300    # Longer timeouts for complex tasks
  max_iterations: 50      # More iterations for complex projects
  allow_network: false    # Restrict network access

safety:
  require_tests: true     # Always run tests
  backup_before_changes: true  # Backup before modifications

logging:
  level: "INFO"           # Reduce log verbosity
  file: "/var/log/a3x.log"
```

### Environment Variables

**Required environment variables:**

```bash
export OPENROUTER_API_KEY="your-production-key"
export A3X_ENVIRONMENT="production"
export A3X_LOG_LEVEL="INFO"
```

**Optional security settings:**

```bash
export A3X_WORKSPACE_ROOT="/opt/projects"  # Restrict workspace
export A3X_MAX_EXECUTION_TIME="3600"       # Max seconds per task
export A3X_ALLOWED_COMMANDS="git,python,pytest"  # Restrict commands
```

## Security Considerations

### API Key Security

1. **Use separate keys for different environments**:
   - Development key for testing
   - Production key for live operations

2. **Key rotation**:
   ```bash
   # Rotate keys periodically
   a3x run --goal "Create script to rotate API keys securely" --config configs/sample.yaml
   ```

3. **Access control**:
   ```bash
   # Restrict key permissions in OpenRouter dashboard
   # Set spending limits and usage restrictions
   ```

### Code Execution Security

1. **Sandbox execution**:
   ```yaml
   execution:
     use_docker: true      # Run in containers
     restrict_network: true # Block network access
     allowed_commands:     # Whitelist commands
       - "python"
       - "pip"
       - "git"
   ```

2. **File system restrictions**:
   ```yaml
   safety:
     allowed_paths:        # Restrict file operations
       - "/opt/projects"
       - "/tmp/a3x-work"
     deny_patterns:        # Block dangerous operations
       - "rm -rf /"
       - "sudo *"
   ```

### Network Security

1. **API endpoint security**:
   ```bash
   # Use HTTPS for all API communications
   # Validate SSL certificates
   # Implement rate limiting
   ```

2. **Webhook security** (if applicable):
   ```bash
   # Validate webhook signatures
   # Use secret tokens for authentication
   # Implement proper request validation
   ```

## Monitoring and Observability

### Logging Configuration

1. **Structured logging**:
   ```yaml
   logging:
     format: "json"        # Structured logs
     level: "INFO"
     file: "/var/log/a3x.log"
     max_size: "100MB"     # Log rotation
   ```

2. **Log aggregation**:
   ```bash
   # Send logs to centralized service
   a3x run --goal "Configure log shipping to Elasticsearch/Logstash" --config configs/sample.yaml
   ```

### Health Checks

1. **Application health**:
   ```bash
   a3x run --goal "Add health check endpoints to the application" --config configs/sample.yaml
   ```

2. **A3X service health**:
   ```bash
   # Create health check script
   a3x run --goal "Create monitoring script for A3X service health" --config configs/sample.yaml
   ```

### Performance Monitoring

1. **Resource monitoring**:
   ```bash
   # Monitor CPU, memory, disk usage
   a3x run --goal "Set up resource monitoring and alerting" --config configs/sample.yaml
   ```

2. **API usage tracking**:
   ```bash
   # Track API costs and usage patterns
   a3x run --goal "Create API usage monitoring dashboard" --config configs/sample.yaml
   ```

## Scaling and Performance

### Horizontal Scaling

1. **Load balancing**:
   ```bash
   a3x run --goal "Configure load balancer for multiple A3X instances" --config configs/sample.yaml
   ```

2. **Session management**:
   ```bash
   a3x run --goal "Implement session management for scaled deployments" --config configs/sample.yaml
   ```

### Resource Optimization

1. **Memory management**:
   ```yaml
   execution:
     max_concurrent_tasks: 3    # Limit concurrent operations
     cleanup_temp_files: true   # Clean up after operations
   ```

2. **Caching strategies**:
   ```bash
   a3x run --goal "Implement caching for frequently used code patterns" --config configs/sample.yaml
   ```

## Backup and Recovery

### Automated Backups

1. **Code backups**:
   ```bash
   a3x run --goal "Set up automated git commits and pushes" --config configs/sample.yaml
   ```

2. **Configuration backups**:
   ```bash
   a3x run --goal "Create automated backup for A3X configurations and seeds" --config configs/sample.yaml
   ```

### Disaster Recovery

1. **Recovery procedures**:
   ```bash
   a3x run --goal "Create disaster recovery playbook for A3X deployment" --config configs/sample.yaml
   ```

2. **Data restoration**:
   ```bash
   a3x run --goal "Create scripts for quick environment restoration" --config configs/sample.yaml
   ```

## Cost Management

### API Cost Optimization

1. **Model selection**:
   ```yaml
   llm:
     model: "x-ai/grok-beta"    # More cost-effective
     max_tokens: 2000          # Reduce token usage
   ```

2. **Usage monitoring**:
   ```bash
   a3x run --goal "Create cost monitoring and alerting system" --config configs/sample.yaml
   ```

3. **Budget controls**:
   ```bash
   # Set spending limits in OpenRouter
   # Use cost-effective models for simple tasks
   # Implement caching for repeated operations
   ```

## Maintenance Procedures

### Regular Maintenance Tasks

1. **Weekly maintenance**:
   ```bash
   # Clean up old logs and temporary files
   a3x run --goal "Perform weekly cleanup and optimization" --config configs/sample.yaml
   ```

2. **Monthly maintenance**:
   ```bash
   # Review and update dependencies
   a3x run --goal "Update dependencies and security patches" --config configs/sample.yaml
   ```

3. **Quarterly review**:
   ```bash
   # Performance review and optimization
   a3x run --goal "Conduct quarterly performance and cost review" --config configs/sample.yaml
   ```

## Troubleshooting Production Issues

### Common Production Issues

1. **Performance degradation**:
   ```bash
   a3x run --goal "Diagnose and fix performance issues in production" --config configs/sample.yaml
   ```

2. **Memory leaks**:
   ```bash
   a3x run --goal "Identify and fix memory leaks in long-running processes" --config configs/sample.yaml
   ```

3. **API failures**:
   ```bash
   a3x run --goal "Debug API connectivity and authentication issues" --config configs/sample.yaml
   ```

### Emergency Procedures

1. **Service recovery**:
   ```bash
   # Quick service restart
   sudo systemctl restart a3x

   # Emergency stop
   sudo systemctl stop a3x
   ```

2. **Data protection**:
   ```bash
   # Emergency backup
   a3x run --goal "Create emergency backup of current state" --config configs/sample.yaml
   ```

## Compliance and Audit

### Audit Logging

1. **Comprehensive audit trail**:
   ```yaml
   logging:
     audit: true            # Enable audit logging
     audit_file: "/var/log/a3x-audit.log"
     log_all_actions: true  # Log all code changes
   ```

2. **Change tracking**:
   ```bash
   a3x run --goal "Implement comprehensive change tracking and reporting" --config configs/sample.yaml
   ```

### Compliance Reporting

1. **Generate compliance reports**:
   ```bash
   a3x run --goal "Create compliance reporting for code changes and deployments" --config configs/sample.yaml
   ```

---

**Remember:** Production deployment requires careful planning and consideration of security, performance, and reliability. Always test thoroughly in staging environments before deploying to production.
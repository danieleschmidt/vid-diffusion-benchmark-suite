# Required Manual Setup

This document outlines the manual setup steps required by repository maintainers due to GitHub App permission limitations during the SDLC implementation.

## GitHub Actions Workflows (REQUIRED)

The following workflow files need to be manually created in `.github/workflows/` from the templates provided in `docs/workflows/examples/`:

### 1. CI/CD Pipeline (`ci.yml`)
Copy from: `docs/workflows/examples/ci.yml`
Purpose: Pull request validation, testing, security scanning

### 2. Deployment Pipeline (`cd.yml`)
Copy from: `docs/workflows/examples/cd.yml`
Purpose: Automated deployment to staging and production

### 3. Security Scanning (`security.yml`)
Copy from: `docs/workflows/examples/security.yml`
Purpose: Comprehensive security scanning and SBOM generation

### 4. Dependency Updates (`dependency-update.yml`)
Copy from: `docs/workflows/examples/dependency-update.yml`
Purpose: Automated dependency management and updates

## Repository Settings

### Branch Protection Rules
Configure for `main` branch:
- Require pull request reviews before merging (2 reviewers)
- Require status checks to pass before merging
- Require branches to be up to date before merging
- Require conversation resolution before merging
- Restrict pushes that create files larger than 100MB

### Required Status Checks
- `ci / test`
- `ci / lint`
- `ci / security-scan`
- `ci / build`

### Repository Secrets
Add the following secrets in repository settings:

```
DOCKER_HUB_USERNAME=<docker_hub_username>
DOCKER_HUB_TOKEN=<docker_hub_token>
NVIDIA_GPU_CLOUD_API_KEY=<ngc_api_key>
WANDB_API_KEY=<weights_and_biases_key>
SLACK_WEBHOOK_URL=<slack_notifications>
```

## Issue and PR Templates

### Issue Templates
Create in `.github/ISSUE_TEMPLATE/`:
- `bug_report.yml` - Bug reports
- `feature_request.yml` - Feature requests  
- `model_request.yml` - New model addition requests

### Pull Request Template
Create `.github/pull_request_template.md` with:
- Description requirements
- Testing checklist
- Breaking changes notice
- Documentation updates

## Security Configuration

### CodeQL Analysis
Enable GitHub Advanced Security:
1. Go to repository Settings > Security & analysis
2. Enable CodeQL analysis
3. Configure for Python, JavaScript, and Docker

### Dependabot
Configure in `.github/dependabot.yml`:
- Python dependency updates (weekly)
- Docker base image updates (weekly)
- GitHub Actions updates (weekly)

### Secret Scanning
Enable secret scanning:
1. Settings > Security & analysis
2. Enable secret scanning
3. Enable push protection

## Deployment Configuration

### Container Registry
Set up GitHub Container Registry:
1. Enable GitHub Packages
2. Configure package permissions
3. Create deployment tokens

### Kubernetes Integration (Optional)
If using Kubernetes deployment:
1. Create namespace: `kubectl create namespace vid-benchmark`
2. Configure RBAC permissions
3. Set up ingress controllers
4. Configure persistent volumes

## Monitoring Integration

### External Monitoring
Set up external monitoring services:
- Uptime monitoring (UptimeRobot, Pingdom)
- Error tracking (Sentry)
- Performance monitoring (New Relic, DataDog)

### Alerting Channels
Configure notification channels:
- Slack workspace integration
- Email notification lists
- PagerDuty integration for critical alerts

## Documentation Hosting

### GitHub Pages
Enable GitHub Pages for documentation:
1. Go to repository Settings > Pages
2. Select source: Deploy from a branch
3. Choose branch: `gh-pages` (created by CI)
4. Custom domain (optional): `docs.vid-benchmark.com`

### Alternative: External Hosting
If using external documentation hosting:
- ReadTheDocs integration
- Netlify deployment
- Vercel integration

## Performance and Analytics

### Performance Monitoring
Set up performance tracking:
- Google Analytics for dashboard usage
- Application Performance Monitoring (APM)
- GPU utilization tracking

### User Analytics
Configure user tracking (if applicable):
- User registration and authentication
- Usage patterns and model preferences
- Performance feedback collection

## Compliance and Governance

### License Compliance
Ensure license compliance:
- Review all model licenses
- Set up license scanning
- Create legal compliance documentation

### Data Privacy
Configure data handling:
- User data collection policies
- Data retention procedures
- GDPR compliance measures (if applicable)

## Quality Gates

### Automated Testing
Ensure all automated tests are configured:
- Unit tests: 90%+ coverage
- Integration tests: Critical paths covered
- Performance tests: Baseline established
- Security tests: Vulnerability scanning

### Code Quality
Configure code quality tools:
- SonarQube/SonarCloud integration
- Code climate analysis
- Technical debt tracking

## Team Access and Permissions

### Repository Access
Configure team permissions:
- Admin: Project leads
- Maintainer: Core team members
- Write: Regular contributors
- Read: External collaborators

### Review Requirements
Set up review policies:
- Code owners file (`.github/CODEOWNERS`)
- Required reviewers by component
- Review assignment automation

## Checklist for Repository Maintainer

- [ ] Create GitHub Actions workflows from templates
- [ ] Configure branch protection rules
- [ ] Add required repository secrets
- [ ] Set up issue and PR templates
- [ ] Enable security features (CodeQL, Dependabot, Secret scanning)
- [ ] Configure container registry access
- [ ] Set up external monitoring and alerting
- [ ] Enable documentation hosting
- [ ] Configure team access permissions
- [ ] Test all automated workflows
- [ ] Document any custom configurations

## Support and Troubleshooting

If you encounter issues during setup:

1. **Workflow Issues**: Check the example files in `docs/workflows/examples/`
2. **Permission Issues**: Verify GitHub App permissions and repository settings
3. **Secret Issues**: Ensure all required secrets are properly configured
4. **Integration Issues**: Check third-party service configurations

For additional support, create an issue with the `setup-help` label.
# Security Policy

## Supported Versions

We actively support security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities by email to: **vid-bench-security@yourdomain.com**

### What to Include

Please include the following information in your report:

- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit the issue

### Response Timeline

- **Acknowledgment**: We will acknowledge receipt of your report within 2 business days
- **Initial Assessment**: We will provide an initial assessment within 5 business days
- **Resolution**: We aim to resolve critical issues within 30 days

### Disclosure Policy

- We follow responsible disclosure practices
- Security fixes will be released as soon as possible
- Credit will be given to security researchers who report vulnerabilities responsibly

## Security Best Practices

### For Users

- Always use the latest version of the benchmark suite
- Keep your dependencies up to date
- Use virtual environments for isolation
- Never commit sensitive data (API keys, tokens) to version control
- Review model weights and datasets from trusted sources only

### For Model Integration

- Validate all input prompts and parameters
- Implement proper resource limits (memory, compute time)
- Sanitize file paths and model loading paths
- Use sandboxed environments for untrusted models
- Monitor resource usage during benchmarking

### For Deployment

- Use HTTPS for all web interfaces
- Implement proper authentication and authorization
- Follow least privilege principles for container permissions
- Regularly scan Docker images for vulnerabilities
- Enable security monitoring and logging

## Known Security Considerations

### Model Safety

- Video diffusion models may generate inappropriate content
- Always review generated videos before public use
- Implement content filtering for production deployments

### Resource Exhaustion

- Large models can consume significant GPU memory and compute
- Implement timeouts and resource limits
- Monitor system resources during benchmarking

### Data Privacy

- Generated videos may contain traces of training data
- Consider privacy implications when benchmarking with sensitive prompts
- Follow applicable data protection regulations

## Security Updates

Security updates will be announced through:

- GitHub Security Advisories
- Release notes with `[SECURITY]` prefix
- Email notifications to maintainers

## Third-Party Dependencies

We regularly audit our dependencies for known vulnerabilities using:

- `safety check` for Python packages
- `bandit` for static code analysis
- Dependabot for automated dependency updates

Report any concerns about third-party dependencies through the same security reporting process.
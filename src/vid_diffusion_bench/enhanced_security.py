"""Enhanced security framework for video diffusion benchmarks."""

import re
import hashlib
import hmac
import secrets
import time
import logging
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, deque
from pathlib import Path
import json
import ipaddress

logger = logging.getLogger(__name__)


@dataclass
class SecurityEvent:
    """Security event record."""
    timestamp: float
    event_type: str
    severity: str  # low, medium, high, critical
    source_ip: Optional[str]
    user_agent: Optional[str]
    description: str
    metadata: Dict[str, Any]


class RateLimiter:
    """Token bucket rate limiter with IP-based tracking."""
    
    def __init__(self, requests_per_minute: int = 60, burst_size: int = 10):
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.tokens_per_second = requests_per_minute / 60.0
        self.buckets = defaultdict(lambda: {'tokens': burst_size, 'last_refill': time.time()})
        
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed for given identifier."""
        now = time.time()
        bucket = self.buckets[identifier]
        
        # Refill tokens based on time elapsed
        time_elapsed = now - bucket['last_refill']
        tokens_to_add = time_elapsed * self.tokens_per_second
        bucket['tokens'] = min(self.burst_size, bucket['tokens'] + tokens_to_add)
        bucket['last_refill'] = now
        
        # Check if request can be served
        if bucket['tokens'] >= 1.0:
            bucket['tokens'] -= 1.0
            return True
        
        return False
        
    def get_retry_after(self, identifier: str) -> float:
        """Get seconds to wait before next request is allowed."""
        bucket = self.buckets[identifier]
        tokens_needed = 1.0 - bucket['tokens']
        return max(0, tokens_needed / self.tokens_per_second)


class InputSanitizer:
    """Comprehensive input sanitization and validation."""
    
    def __init__(self):
        # Dangerous patterns to block
        self.dangerous_patterns = [
            # Script injection
            r'<script[^>]*>.*?</script>',
            r'javascript:',
            r'vbscript:',
            r'data:text/html',
            r'data:application/javascript',
            
            # Command injection
            r';\s*(?:rm|del|format|shutdown|reboot)',
            r'\|\s*(?:cat|type|more|less)',
            r'`[^`]*`',  # Backticks
            r'\$\([^)]*\)',  # Command substitution
            
            # Path traversal
            r'\.\./',
            r'\.\.\\',
            r'/etc/passwd',
            r'/proc/',
            r'C:\\Windows',
            
            # Code execution
            r'exec\s*\(',
            r'eval\s*\(',
            r'__import__',
            r'subprocess',
            r'os\.system',
            
            # SQL injection basic patterns
            r"'\s*(?:or|and)\s*'",
            r'union\s+select',
            r'drop\s+table',
            
            # File inclusion
            r'file://',
            r'ftp://',
            r'\\\\[a-zA-Z0-9]',
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE | re.DOTALL) 
                                 for pattern in self.dangerous_patterns]
        
        # Maximum lengths
        self.max_lengths = {
            'prompt': 2000,
            'model_name': 100,
            'file_path': 500
        }
        
    def sanitize_prompt(self, prompt: str) -> Tuple[str, List[str]]:
        """Sanitize text prompt and return warnings."""
        if not isinstance(prompt, str):
            raise ValueError("Prompt must be a string")
            
        original = prompt
        warnings = []
        
        # Length check
        if len(prompt) > self.max_lengths['prompt']:
            prompt = prompt[:self.max_lengths['prompt']]
            warnings.append(f"Prompt truncated to {self.max_lengths['prompt']} characters")
            
        # Remove null bytes
        if '\x00' in prompt:
            prompt = prompt.replace('\x00', '')
            warnings.append("Null bytes removed")
            
        # Check for dangerous patterns
        for pattern in self.compiled_patterns:
            if pattern.search(prompt):
                warnings.append(f"Potentially dangerous pattern detected: {pattern.pattern[:50]}...")
                # For now, just warn - in production might block or sanitize further
                
        # Basic HTML entity encoding for special characters
        replacements = {
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#x27;',
            '&': '&amp;'
        }
        
        for char, replacement in replacements.items():
            if char in prompt:
                prompt = prompt.replace(char, replacement)
                if char in original:
                    warnings.append(f"HTML character '{char}' encoded")
                    
        # Normalize whitespace
        prompt = ' '.join(prompt.split())
        
        return prompt, warnings
        
    def sanitize_file_path(self, path: str) -> Tuple[str, List[str]]:
        """Sanitize file path."""
        if not isinstance(path, str):
            raise ValueError("Path must be a string")
            
        warnings = []
        
        # Length check
        if len(path) > self.max_lengths['file_path']:
            raise ValueError(f"Path too long: {len(path)} > {self.max_lengths['file_path']}")
            
        # Remove dangerous characters
        dangerous_chars = ['<', '>', '|', '*', '?', '"']
        for char in dangerous_chars:
            if char in path:
                path = path.replace(char, '_')
                warnings.append(f"Dangerous character '{char}' replaced")
                
        # Block path traversal attempts
        if '..' in path:
            raise ValueError("Path traversal detected")
            
        # Normalize path separators
        path = path.replace('\\', '/')
        
        # Remove multiple consecutive slashes
        while '//' in path:
            path = path.replace('//', '/')
            
        return path, warnings
        
    def validate_model_name(self, model_name: str) -> bool:
        """Validate model name format."""
        if not isinstance(model_name, str):
            return False
            
        if len(model_name) > self.max_lengths['model_name']:
            return False
            
        # Allow alphanumeric, hyphens, underscores, dots
        if not re.match(r'^[a-zA-Z0-9._-]+$', model_name):
            return False
            
        return True


class AccessControl:
    """Role-based access control system."""
    
    def __init__(self):
        self.roles = {
            'admin': {
                'permissions': ['*'],  # All permissions
                'description': 'Full system access'
            },
            'researcher': {
                'permissions': [
                    'benchmark.read', 'benchmark.create', 'benchmark.compare',
                    'model.list', 'model.evaluate', 'metrics.view',
                    'research.analyze', 'research.export'
                ],
                'description': 'Research and benchmarking access'
            },
            'viewer': {
                'permissions': [
                    'benchmark.read', 'model.list', 'metrics.view'
                ],
                'description': 'Read-only access'
            },
            'demo': {
                'permissions': [
                    'model.evaluate', 'benchmark.read'
                ],
                'description': 'Limited demo access',
                'restrictions': {
                    'max_prompts_per_request': 3,
                    'max_frames': 16,
                    'allowed_models': ['mock-fast', 'mock-efficient']
                }
            }
        }
        
        self.user_roles = {}  # user_id -> role_name
        self.api_keys = {}    # api_key -> user_id
        
    def create_api_key(self, user_id: str, role: str) -> str:
        """Create API key for user with specific role."""
        if role not in self.roles:
            raise ValueError(f"Invalid role: {role}")
            
        api_key = secrets.token_urlsafe(32)
        self.api_keys[api_key] = user_id
        self.user_roles[user_id] = role
        
        logger.info(f"Created API key for user {user_id} with role {role}")
        return api_key
        
    def authenticate(self, api_key: str) -> Optional[Tuple[str, str]]:
        """Authenticate API key and return (user_id, role)."""
        user_id = self.api_keys.get(api_key)
        if not user_id:
            return None
            
        role = self.user_roles.get(user_id)
        if not role:
            return None
            
        return user_id, role
        
    def authorize(self, user_id: str, permission: str) -> bool:
        """Check if user has required permission."""
        role = self.user_roles.get(user_id)
        if not role:
            return False
            
        role_info = self.roles.get(role)
        if not role_info:
            return False
            
        permissions = role_info['permissions']
        
        # Check wildcard permission
        if '*' in permissions:
            return True
            
        # Check exact permission match
        if permission in permissions:
            return True
            
        # Check prefix match (e.g., 'benchmark.*' covers 'benchmark.read')
        for perm in permissions:
            if perm.endswith('*'):
                prefix = perm[:-1]
                if permission.startswith(prefix):
                    return True
                    
        return False
        
    def get_user_restrictions(self, user_id: str) -> Dict[str, Any]:
        """Get user-specific restrictions."""
        role = self.user_roles.get(user_id)
        if not role:
            return {}
            
        role_info = self.roles.get(role, {})
        return role_info.get('restrictions', {})


class SecurityAuditor:
    """Security auditing and threat detection."""
    
    def __init__(self, max_events: int = 10000):
        self.events = deque(maxlen=max_events)
        self.threat_patterns = {
            'brute_force': {
                'pattern': 'multiple_auth_failures',
                'threshold': 10,
                'window_minutes': 5
            },
            'prompt_injection': {
                'pattern': 'dangerous_prompt_pattern',
                'threshold': 3,
                'window_minutes': 10
            },
            'rate_limit_abuse': {
                'pattern': 'rate_limit_exceeded',
                'threshold': 20,
                'window_minutes': 5
            }
        }
        
    def log_event(self, event_type: str, severity: str, description: str, 
                  source_ip: str = None, user_agent: str = None, **metadata):
        """Log security event."""
        event = SecurityEvent(
            timestamp=time.time(),
            event_type=event_type,
            severity=severity,
            source_ip=source_ip,
            user_agent=user_agent,
            description=description,
            metadata=metadata
        )
        
        self.events.append(event)
        
        # Log to system logger
        log_level = {
            'low': logging.INFO,
            'medium': logging.WARNING,
            'high': logging.ERROR,
            'critical': logging.CRITICAL
        }.get(severity, logging.INFO)
        
        logger.log(log_level, f"Security event: {description}", extra={
            'event_type': event_type,
            'severity': severity,
            'source_ip': source_ip,
            **metadata
        })
        
        # Check for threat patterns
        self._check_threat_patterns(event)
        
    def _check_threat_patterns(self, new_event: SecurityEvent):
        """Check for threat patterns and trigger alerts."""
        now = time.time()
        
        for threat_name, threat_config in self.threat_patterns.items():
            pattern = threat_config['pattern']
            threshold = threat_config['threshold']
            window_seconds = threat_config['window_minutes'] * 60
            
            # Count matching events in time window
            matching_events = [
                event for event in self.events
                if (now - event.timestamp <= window_seconds and 
                    self._event_matches_pattern(event, pattern))
            ]
            
            if len(matching_events) >= threshold:
                self._trigger_threat_alert(threat_name, matching_events)
                
    def _event_matches_pattern(self, event: SecurityEvent, pattern: str) -> bool:
        """Check if event matches threat pattern."""
        pattern_mappings = {
            'multiple_auth_failures': lambda e: e.event_type == 'auth_failure',
            'dangerous_prompt_pattern': lambda e: e.event_type == 'dangerous_prompt',
            'rate_limit_exceeded': lambda e: e.event_type == 'rate_limit_exceeded'
        }
        
        matcher = pattern_mappings.get(pattern)
        return matcher(event) if matcher else False
        
    def _trigger_threat_alert(self, threat_name: str, events: List[SecurityEvent]):
        """Trigger threat detection alert."""
        source_ips = list(set(e.source_ip for e in events if e.source_ip))
        
        alert_event = SecurityEvent(
            timestamp=time.time(),
            event_type='threat_detected',
            severity='critical',
            source_ip=source_ips[0] if source_ips else None,
            user_agent=None,
            description=f"Threat pattern detected: {threat_name}",
            metadata={
                'threat_name': threat_name,
                'event_count': len(events),
                'source_ips': source_ips,
                'time_window': events[-1].timestamp - events[0].timestamp
            }
        )
        
        self.events.append(alert_event)
        logger.critical(f"THREAT DETECTED: {threat_name} with {len(events)} events")
        
    def get_security_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate security report for specified time period."""
        cutoff = time.time() - (hours * 3600)
        recent_events = [e for e in self.events if e.timestamp >= cutoff]
        
        # Count events by type and severity
        event_counts = defaultdict(int)
        severity_counts = defaultdict(int)
        ip_counts = defaultdict(int)
        
        for event in recent_events:
            event_counts[event.event_type] += 1
            severity_counts[event.severity] += 1
            if event.source_ip:
                ip_counts[event.source_ip] += 1
                
        # Identify top threat sources
        top_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Calculate threat level
        threat_score = (
            severity_counts['critical'] * 10 +
            severity_counts['high'] * 5 +
            severity_counts['medium'] * 2 +
            severity_counts['low'] * 1
        )
        
        threat_level = 'low'
        if threat_score > 100:
            threat_level = 'critical'
        elif threat_score > 50:
            threat_level = 'high'
        elif threat_score > 20:
            threat_level = 'medium'
            
        return {
            'period_hours': hours,
            'total_events': len(recent_events),
            'threat_level': threat_level,
            'threat_score': threat_score,
            'events_by_type': dict(event_counts),
            'events_by_severity': dict(severity_counts),
            'top_source_ips': top_ips,
            'recent_critical_events': [
                {
                    'timestamp': e.timestamp,
                    'type': e.event_type,
                    'description': e.description,
                    'source_ip': e.source_ip
                }
                for e in recent_events 
                if e.severity == 'critical'
            ][-10:]  # Last 10 critical events
        }


class IPBlocklist:
    """IP address blocking and reputation management."""
    
    def __init__(self):
        self.blocked_ips = set()
        self.blocked_ranges = []  # List of (network, reason, expiry) tuples
        self.reputation_scores = defaultdict(int)  # IP -> score (lower is worse)
        
    def block_ip(self, ip: str, reason: str, duration_minutes: int = None):
        """Block specific IP address."""
        self.blocked_ips.add(ip)
        expiry = time.time() + (duration_minutes * 60) if duration_minutes else None
        
        logger.warning(f"Blocked IP {ip}: {reason}")
        
    def block_range(self, cidr: str, reason: str, duration_minutes: int = None):
        """Block IP range in CIDR notation."""
        try:
            network = ipaddress.ip_network(cidr, strict=False)
            expiry = time.time() + (duration_minutes * 60) if duration_minutes else None
            self.blocked_ranges.append((network, reason, expiry))
            
            logger.warning(f"Blocked IP range {cidr}: {reason}")
        except ValueError as e:
            logger.error(f"Invalid CIDR range {cidr}: {e}")
            
    def is_blocked(self, ip: str) -> Tuple[bool, Optional[str]]:
        """Check if IP is blocked."""
        # Check direct IP blocks
        if ip in self.blocked_ips:
            return True, "IP directly blocked"
            
        # Check range blocks
        try:
            ip_obj = ipaddress.ip_address(ip)
            current_time = time.time()
            
            for network, reason, expiry in self.blocked_ranges:
                # Remove expired blocks
                if expiry and current_time > expiry:
                    continue
                    
                if ip_obj in network:
                    return True, f"IP in blocked range: {reason}"
                    
        except ValueError:
            logger.warning(f"Invalid IP address format: {ip}")
            return True, "Invalid IP format"
            
        return False, None
        
    def update_reputation(self, ip: str, delta: int):
        """Update IP reputation score."""
        self.reputation_scores[ip] += delta
        
        # Auto-block IPs with very low reputation
        if self.reputation_scores[ip] <= -100:
            self.block_ip(ip, "Low reputation score", duration_minutes=60)
            
    def get_reputation(self, ip: str) -> int:
        """Get IP reputation score."""
        return self.reputation_scores.get(ip, 0)
        
    def cleanup_expired_blocks(self):
        """Remove expired IP blocks."""
        current_time = time.time()
        
        # Clean up range blocks
        self.blocked_ranges = [
            (network, reason, expiry) for network, reason, expiry in self.blocked_ranges
            if not expiry or expiry > current_time
        ]


class BenchmarkSecurityManager:
    """Comprehensive security management for benchmark system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Initialize components
        self.rate_limiter = RateLimiter(
            requests_per_minute=self.config.get('rate_limit_rpm', 60),
            burst_size=self.config.get('rate_limit_burst', 10)
        )
        self.sanitizer = InputSanitizer()
        self.access_control = AccessControl()
        self.auditor = SecurityAuditor()
        self.ip_blocklist = IPBlocklist()
        
        # Load default blocked IPs/ranges
        self._load_default_blocks()
        
    def _load_default_blocks(self):
        """Load default IP blocks."""
        # Block common attack sources (example ranges)
        default_blocks = [
            ('10.0.0.0/8', 'Private network'),  # Usually shouldn't access public APIs
            ('192.168.0.0/16', 'Private network'),
            ('172.16.0.0/12', 'Private network'),
        ]
        
        for cidr, reason in default_blocks:
            if self.config.get('block_private_networks', False):
                self.ip_blocklist.block_range(cidr, reason)
                
    def validate_request(self, 
                        source_ip: str,
                        api_key: str = None,
                        model_name: str = None,
                        prompts: List[str] = None,
                        **params) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Comprehensive request validation."""
        
        warnings = []
        metadata = {}
        
        # IP blocking check
        is_blocked, block_reason = self.ip_blocklist.is_blocked(source_ip)
        if is_blocked:
            self.auditor.log_event(
                'blocked_ip_attempt', 'high',
                f"Blocked IP attempted access: {block_reason}",
                source_ip=source_ip
            )
            return False, [f"Access denied: {block_reason}"], metadata
            
        # Rate limiting
        if not self.rate_limiter.is_allowed(source_ip):
            retry_after = self.rate_limiter.get_retry_after(source_ip)
            self.auditor.log_event(
                'rate_limit_exceeded', 'medium',
                f"Rate limit exceeded for IP {source_ip}",
                source_ip=source_ip,
                retry_after=retry_after
            )
            return False, [f"Rate limit exceeded. Retry after {retry_after:.1f} seconds"], metadata
            
        # Authentication
        user_id, role = None, None
        if api_key:
            auth_result = self.access_control.authenticate(api_key)
            if not auth_result:
                self.auditor.log_event(
                    'auth_failure', 'medium',
                    f"Invalid API key from {source_ip}",
                    source_ip=source_ip
                )
                self.ip_blocklist.update_reputation(source_ip, -10)
                return False, ["Invalid API key"], metadata
            
            user_id, role = auth_result
            metadata['user_id'] = user_id
            metadata['role'] = role
            
        # Input sanitization
        if model_name:
            if not self.sanitizer.validate_model_name(model_name):
                self.auditor.log_event(
                    'invalid_model_name', 'medium',
                    f"Invalid model name: {model_name}",
                    source_ip=source_ip,
                    model_name=model_name
                )
                return False, ["Invalid model name"], metadata
                
        if prompts:
            sanitized_prompts = []
            for i, prompt in enumerate(prompts):
                try:
                    clean_prompt, prompt_warnings = self.sanitizer.sanitize_prompt(prompt)
                    sanitized_prompts.append(clean_prompt)
                    warnings.extend([f"Prompt {i}: {w}" for w in prompt_warnings])
                    
                    # Check for dangerous patterns
                    if any('dangerous' in w.lower() for w in prompt_warnings):
                        self.auditor.log_event(
                            'dangerous_prompt', 'high',
                            f"Dangerous pattern in prompt from {source_ip}",
                            source_ip=source_ip,
                            prompt_preview=prompt[:100]
                        )
                        self.ip_blocklist.update_reputation(source_ip, -20)
                        
                except ValueError as e:
                    return False, [f"Invalid prompt {i}: {str(e)}"], metadata
                    
            metadata['sanitized_prompts'] = sanitized_prompts
            
        # Authorization (if authenticated)
        if user_id:
            required_permission = 'benchmark.create' if prompts else 'benchmark.read'
            if not self.access_control.authorize(user_id, required_permission):
                self.auditor.log_event(
                    'authorization_failure', 'medium',
                    f"User {user_id} lacks permission {required_permission}",
                    source_ip=source_ip,
                    user_id=user_id,
                    permission=required_permission
                )
                return False, [f"Permission denied: {required_permission}"], metadata
                
            # Check user restrictions
            restrictions = self.access_control.get_user_restrictions(user_id)
            if restrictions:
                if prompts and len(prompts) > restrictions.get('max_prompts_per_request', float('inf')):
                    return False, [f"Too many prompts. Max: {restrictions['max_prompts_per_request']}"], metadata
                    
                if model_name:
                    allowed_models = restrictions.get('allowed_models')
                    if allowed_models and model_name not in allowed_models:
                        return False, [f"Model not allowed. Allowed: {allowed_models}"], metadata
                        
        # Update IP reputation for successful validation
        self.ip_blocklist.update_reputation(source_ip, 1)
        
        # Log successful validation
        self.auditor.log_event(
            'request_validated', 'low',
            f"Request validated for {source_ip}",
            source_ip=source_ip,
            user_id=user_id,
            model_name=model_name
        )
        
        return True, warnings, metadata
        
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            'timestamp': time.time(),
            'blocked_ips': len(self.ip_blocklist.blocked_ips),
            'blocked_ranges': len(self.ip_blocklist.blocked_ranges),
            'total_users': len(self.access_control.user_roles),
            'active_api_keys': len(self.access_control.api_keys),
            'recent_events': len([e for e in self.auditor.events if time.time() - e.timestamp < 3600]),
            'security_report': self.auditor.get_security_report(hours=24)
        }
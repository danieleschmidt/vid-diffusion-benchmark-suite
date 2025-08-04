"""Input sanitization utilities."""

import re
import html
import unicodedata
from typing import Any, Dict, List, Union
from pathlib import Path

from ..monitoring.logging import get_structured_logger

logger = get_structured_logger(__name__)


def sanitize_input(value: Any, max_length: int = 1000) -> str:
    """Sanitize general input value.
    
    Args:
        value: Input value to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
    """
    if value is None:
        return ""
        
    # Convert to string
    text = str(value)
    
    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length]
        logger.warning(f"Input truncated to {max_length} characters")
        
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Normalize unicode
    text = unicodedata.normalize('NFKC', text)
    
    # Remove control characters except newlines, tabs, and carriage returns
    text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t\r')
    
    # HTML escape dangerous characters
    text = html.escape(text, quote=True)
    
    return text.strip()


def sanitize_filename(filename: str, max_length: int = 255) -> str:
    """Sanitize filename for safe filesystem usage.
    
    Args:
        filename: Original filename
        max_length: Maximum filename length
        
    Returns:
        Sanitized filename
    """
    if not filename:
        return "untitled"
        
    # Convert to string and normalize
    filename = str(filename)
    filename = unicodedata.normalize('NFKC', filename)
    
    # Remove path separators and dangerous characters
    dangerous_chars = r'[<>:"/\\|?*\x00-\x1f]'
    filename = re.sub(dangerous_chars, '_', filename)
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('. ')
    
    # Handle reserved Windows names
    reserved_names = {
        'CON', 'PRN', 'AUX', 'NUL',
        'COM1', 'COM2', 'COM3', 'COM4', 'COM5', 'COM6', 'COM7', 'COM8', 'COM9',
        'LPT1', 'LPT2', 'LPT3', 'LPT4', 'LPT5', 'LPT6', 'LPT7', 'LPT8', 'LPT9'
    }
    
    name_part = Path(filename).stem
    if name_part.upper() in reserved_names:
        filename = f"_{filename}"
        
    # Truncate if too long
    if len(filename) > max_length:
        name = Path(filename).stem
        ext = Path(filename).suffix
        available_length = max_length - len(ext)
        filename = name[:available_length] + ext
        
    # Ensure we have a filename
    if not filename or filename == '.':
        filename = "untitled"
        
    return filename


def sanitize_path(path: str, base_dir: str = None) -> str:
    """Sanitize file path to prevent directory traversal.
    
    Args:
        path: File path to sanitize
        base_dir: Base directory to restrict to
        
    Returns:
        Sanitized path
    """
    if not path:
        return ""
        
    # Convert to Path object for normalization
    path_obj = Path(path)
    
    # Resolve path to eliminate .. and .
    try:
        resolved_path = path_obj.resolve()
    except (OSError, ValueError):
        # If resolution fails, create a safe path
        parts = [sanitize_filename(part) for part in path_obj.parts if part not in ('..', '.')]
        resolved_path = Path(*parts) if parts else Path("untitled")
        
    # If base directory is specified, ensure path is within it
    if base_dir:
        base_path = Path(base_dir).resolve()
        try:
            resolved_path.relative_to(base_path)
        except ValueError:
            # Path is outside base directory, create safe path within base
            filename = sanitize_filename(resolved_path.name)
            resolved_path = base_path / filename
            
    return str(resolved_path)


def sanitize_json(data: Any, max_depth: int = 10, max_items: int = 1000) -> Any:
    """Sanitize JSON data structure.
    
    Args:
        data: JSON data to sanitize
        max_depth: Maximum nesting depth
        max_items: Maximum number of items in collections
        
    Returns:
        Sanitized data structure
    """
    def _sanitize_recursive(obj: Any, depth: int = 0) -> Any:
        if depth > max_depth:
            return "[MAX_DEPTH_EXCEEDED]"
            
        if isinstance(obj, dict):
            if len(obj) > max_items:
                logger.warning(f"Dictionary truncated to {max_items} items")
                obj = dict(list(obj.items())[:max_items])
                
            return {
                sanitize_input(str(k), 100): _sanitize_recursive(v, depth + 1)
                for k, v in obj.items()
            }
            
        elif isinstance(obj, list):
            if len(obj) > max_items:
                logger.warning(f"List truncated to {max_items} items")
                obj = obj[:max_items]
                
            return [_sanitize_recursive(item, depth + 1) for item in obj]
            
        elif isinstance(obj, str):
            return sanitize_input(obj, 10000)  # Allow longer strings in JSON
            
        elif isinstance(obj, (int, float, bool)) or obj is None:
            return obj
            
        else:
            # Convert unknown types to string and sanitize
            return sanitize_input(str(obj), 1000)
            
    return _sanitize_recursive(data)


def sanitize_sql_identifier(identifier: str) -> str:
    """Sanitize SQL identifier (table name, column name, etc.).
    
    Args:
        identifier: SQL identifier to sanitize
        
    Returns:
        Sanitized identifier
    """
    if not identifier:
        return ""
        
    # Remove all non-alphanumeric characters except underscores
    sanitized = re.sub(r'[^\w]', '_', identifier)
    
    # Ensure it starts with a letter or underscore
    if sanitized and sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
        
    # Truncate to reasonable length
    sanitized = sanitized[:64]
    
    # Ensure we have something
    if not sanitized:
        sanitized = "unnamed"
        
    return sanitized


def sanitize_url(url: str) -> str:
    """Sanitize URL to prevent malicious redirects.
    
    Args:
        url: URL to sanitize
        
    Returns:
        Sanitized URL
    """
    if not url:
        return ""
        
    # Basic URL validation
    url = url.strip()
    
    # Remove dangerous protocols
    dangerous_protocols = ['javascript:', 'data:', 'vbscript:', 'file:']
    url_lower = url.lower()
    
    for protocol in dangerous_protocols:
        if url_lower.startswith(protocol):
            logger.warning(f"Blocked dangerous URL protocol: {protocol}")
            return ""
            
    # Ensure http/https for external URLs
    if url.startswith('//'):
        url = 'https:' + url
    elif not url.startswith(('http://', 'https://', '/')):
        url = 'https://' + url
        
    # Remove control characters
    url = ''.join(char for char in url if ord(char) >= 32)
    
    # Truncate long URLs
    if len(url) > 2048:
        url = url[:2048]
        
    return url


def sanitize_html(html_content: str, allowed_tags: List[str] = None) -> str:
    """Sanitize HTML content to prevent XSS.
    
    Args:
        html_content: HTML content to sanitize
        allowed_tags: List of allowed HTML tags
        
    Returns:
        Sanitized HTML
    """
    if not html_content:
        return ""
        
    # Default allowed tags (very restrictive)
    if allowed_tags is None:
        allowed_tags = ['p', 'br', 'strong', 'em', 'ul', 'ol', 'li']
        
    # Remove all script tags and their content
    html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove all style tags and their content
    html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove event handlers
    html_content = re.sub(r'\son\w+="[^"]*"', '', html_content, flags=re.IGNORECASE)
    html_content = re.sub(r"\son\w+='[^']*'", '', html_content, flags=re.IGNORECASE)
    
    # Remove javascript: URLs
    html_content = re.sub(r'href\s*=\s*["\']javascript:[^"\']*["\']', '', html_content, flags=re.IGNORECASE)
    
    # If no allowed tags, escape everything
    if not allowed_tags:
        return html.escape(html_content)
        
    # Remove disallowed tags (simple approach - for production use a proper HTML sanitizer)
    allowed_pattern = '|'.join(allowed_tags)
    disallowed_tags = re.sub(f'<(?!/?(?:{allowed_pattern})\\b)[^>]*>', '', html_content, flags=re.IGNORECASE)
    
    return disallowed_tags


def sanitize_command_args(args: List[str]) -> List[str]:
    """Sanitize command line arguments to prevent injection.
    
    Args:
        args: List of command arguments
        
    Returns:
        Sanitized arguments
    """
    sanitized_args = []
    
    for arg in args:
        if not isinstance(arg, str):
            arg = str(arg)
            
        # Remove null bytes
        arg = arg.replace('\x00', '')
        
        # Remove shell metacharacters
        dangerous_chars = '|&;()<>$`\\'
        for char in dangerous_chars:
            arg = arg.replace(char, '')
            
        # Remove leading dashes to prevent option injection
        arg = arg.lstrip('-')
        
        # Truncate long arguments
        if len(arg) > 1000:
            arg = arg[:1000]
            
        sanitized_args.append(arg)
        
    return sanitized_args


class InputSanitizer:
    """Configurable input sanitizer."""
    
    def __init__(self, 
                 max_string_length: int = 1000,
                 max_list_items: int = 1000,
                 max_dict_items: int = 1000,
                 max_depth: int = 10):
        """Initialize sanitizer with limits.
        
        Args:
            max_string_length: Maximum string length
            max_list_items: Maximum list items
            max_dict_items: Maximum dictionary items
            max_depth: Maximum nesting depth
        """
        self.max_string_length = max_string_length
        self.max_list_items = max_list_items
        self.max_dict_items = max_dict_items
        self.max_depth = max_depth
        
    def sanitize(self, data: Any) -> Any:
        """Sanitize arbitrary data structure.
        
        Args:
            data: Data to sanitize
            
        Returns:
            Sanitized data
        """
        return self._sanitize_recursive(data, 0)
        
    def _sanitize_recursive(self, obj: Any, depth: int) -> Any:
        """Recursively sanitize data structure."""
        if depth > self.max_depth:
            return "[MAX_DEPTH_EXCEEDED]"
            
        if isinstance(obj, str):
            return sanitize_input(obj, self.max_string_length)
            
        elif isinstance(obj, dict):
            if len(obj) > self.max_dict_items:
                obj = dict(list(obj.items())[:self.max_dict_items])
                
            return {
                sanitize_input(str(k), 100): self._sanitize_recursive(v, depth + 1)
                for k, v in obj.items()
            }
            
        elif isinstance(obj, list):
            if len(obj) > self.max_list_items:
                obj = obj[:self.max_list_items]
                
            return [self._sanitize_recursive(item, depth + 1) for item in obj]
            
        elif isinstance(obj, (int, float, bool)) or obj is None:
            return obj
            
        else:
            return sanitize_input(str(obj), self.max_string_length)


# Global sanitizer instance
default_sanitizer = InputSanitizer()
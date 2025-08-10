"""Comprehensive internationalization (i18n) framework for video diffusion benchmarking.

Multi-language support with advanced features including RTL languages,
cultural adaptations, locale-specific formatting, and dynamic translation loading.
"""

import logging
import json
import re
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import datetime
import threading
from collections import defaultdict

# Third-party imports for advanced i18n features
try:
    import babel
    from babel import Locale, dates, numbers, units
    from babel.messages import Catalog
    from babel.support import Translations
    BABEL_AVAILABLE = True
except ImportError:
    BABEL_AVAILABLE = False

try:
    import icu
    ICU_AVAILABLE = True
except ImportError:
    ICU_AVAILABLE = False

logger = logging.getLogger(__name__)


class LanguageCode(Enum):
    """Supported language codes following ISO 639-1 standard."""
    ENGLISH = "en"
    SPANISH = "es" 
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE_SIMPLIFIED = "zh-CN"
    CHINESE_TRADITIONAL = "zh-TW"
    KOREAN = "ko"
    ITALIAN = "it"
    PORTUGUESE = "pt"
    RUSSIAN = "ru"
    ARABIC = "ar"
    HEBREW = "he"
    HINDI = "hi"
    DUTCH = "nl"
    SWEDISH = "sv"
    NORWEGIAN = "no"
    FINNISH = "fi"
    DANISH = "da"
    POLISH = "pl"


class TextDirection(Enum):
    """Text direction for UI layout."""
    LTR = "ltr"  # Left-to-right
    RTL = "rtl"  # Right-to-left


@dataclass
class LocaleConfig:
    """Configuration for a specific locale."""
    language_code: str
    country_code: Optional[str] = None
    display_name: str = ""
    native_name: str = ""
    text_direction: TextDirection = TextDirection.LTR
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    decimal_separator: str = "."
    thousand_separator: str = ","
    currency_code: str = "USD"
    currency_symbol: str = "$"
    translation_coverage: float = 1.0
    fallback_locales: List[str] = field(default_factory=list)


@dataclass
class TranslationContext:
    """Context information for translations."""
    domain: str = "benchmark"  # Translation domain (e.g., "ui", "errors", "metrics")
    context: Optional[str] = None  # Specific context for disambiguation
    pluralization: Optional[str] = None  # Plural form context
    variables: Dict[str, Any] = field(default_factory=dict)  # Template variables
    gender: Optional[str] = None  # Gender for gendered languages
    formality: Optional[str] = None  # Formal/informal register


class TranslationManager:
    """Advanced translation management with caching and fallback support."""
    
    def __init__(self, base_path: Path = None, default_locale: str = "en"):
        self.base_path = base_path or Path(__file__).parent / "locales"
        self.default_locale = default_locale
        self.current_locale = default_locale
        
        # Translation storage
        self.translations: Dict[str, Dict[str, Any]] = {}
        self.locale_configs: Dict[str, LocaleConfig] = {}
        
        # Caching
        self.translation_cache: Dict[str, str] = {}
        self.cache_lock = threading.RLock()
        
        # Performance tracking
        self.translation_stats = defaultdict(int)
        
        # Initialize supported locales
        self._initialize_supported_locales()
        self._load_translations()
    
    def _initialize_supported_locales(self):
        """Initialize configuration for supported locales."""
        locale_configs = {
            "en": LocaleConfig(
                language_code="en",
                country_code="US",
                display_name="English",
                native_name="English",
                text_direction=TextDirection.LTR,
                fallback_locales=[]
            ),
            "es": LocaleConfig(
                language_code="es",
                country_code="ES", 
                display_name="Spanish",
                native_name="Español",
                text_direction=TextDirection.LTR,
                fallback_locales=["en"]
            ),
            "fr": LocaleConfig(
                language_code="fr",
                country_code="FR",
                display_name="French", 
                native_name="Français",
                text_direction=TextDirection.LTR,
                fallback_locales=["en"]
            ),
            "de": LocaleConfig(
                language_code="de",
                country_code="DE",
                display_name="German",
                native_name="Deutsch", 
                text_direction=TextDirection.LTR,
                fallback_locales=["en"]
            ),
            "ja": LocaleConfig(
                language_code="ja",
                country_code="JP",
                display_name="Japanese",
                native_name="日本語",
                text_direction=TextDirection.LTR,
                date_format="%Y年%m月%d日",
                fallback_locales=["en"]
            ),
            "zh-CN": LocaleConfig(
                language_code="zh",
                country_code="CN", 
                display_name="Chinese (Simplified)",
                native_name="简体中文",
                text_direction=TextDirection.LTR,
                fallback_locales=["en"]
            ),
            "zh-TW": LocaleConfig(
                language_code="zh",
                country_code="TW",
                display_name="Chinese (Traditional)",
                native_name="繁體中文", 
                text_direction=TextDirection.LTR,
                fallback_locales=["zh-CN", "en"]
            ),
            "ko": LocaleConfig(
                language_code="ko",
                country_code="KR",
                display_name="Korean",
                native_name="한국어",
                text_direction=TextDirection.LTR,
                fallback_locales=["en"]
            ),
            "ar": LocaleConfig(
                language_code="ar",
                country_code="SA",
                display_name="Arabic",
                native_name="العربية",
                text_direction=TextDirection.RTL,
                fallback_locales=["en"]
            ),
            "he": LocaleConfig(
                language_code="he", 
                country_code="IL",
                display_name="Hebrew",
                native_name="עברית",
                text_direction=TextDirection.RTL,
                fallback_locales=["en"]
            ),
            "ru": LocaleConfig(
                language_code="ru",
                country_code="RU",
                display_name="Russian", 
                native_name="Русский",
                text_direction=TextDirection.LTR,
                fallback_locales=["en"]
            ),
            "hi": LocaleConfig(
                language_code="hi",
                country_code="IN",
                display_name="Hindi",
                native_name="हिन्दी", 
                text_direction=TextDirection.LTR,
                fallback_locales=["en"]
            )
        }
        
        self.locale_configs.update(locale_configs)
    
    def _load_translations(self):
        """Load translation files for all supported locales."""
        for locale_code in self.locale_configs.keys():
            self._load_locale_translations(locale_code)
    
    def _load_locale_translations(self, locale: str):
        """Load translations for a specific locale."""
        locale_dir = self.base_path / locale
        
        if not locale_dir.exists():
            logger.warning(f"Translation directory not found for locale {locale}: {locale_dir}")
            self.translations[locale] = {}
            return
        
        translations = {}
        
        # Load JSON translation files
        for json_file in locale_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    domain_translations = json.load(f)
                    domain = json_file.stem
                    translations[domain] = domain_translations
                    
            except Exception as e:
                logger.error(f"Failed to load translations from {json_file}: {e}")
        
        # Load gettext .po files if available
        if BABEL_AVAILABLE:
            po_file = locale_dir / "LC_MESSAGES" / "messages.po"
            if po_file.exists():
                try:
                    with open(po_file, 'rb') as f:
                        catalog = Catalog()
                        # Would parse .po file content here
                        # For now, just log that it's available
                        logger.debug(f"Found gettext file for {locale}: {po_file}")
                except Exception as e:
                    logger.error(f"Failed to load gettext file {po_file}: {e}")
        
        self.translations[locale] = translations
        logger.info(f"Loaded translations for locale {locale}: {len(translations)} domains")
    
    def set_locale(self, locale: str) -> bool:
        """Set the current active locale."""
        if locale not in self.locale_configs:
            logger.error(f"Unsupported locale: {locale}")
            return False
        
        self.current_locale = locale
        
        # Clear translation cache when locale changes
        with self.cache_lock:
            self.translation_cache.clear()
        
        logger.info(f"Active locale set to: {locale}")
        return True
    
    def get_supported_locales(self) -> List[str]:
        """Get list of supported locale codes."""
        return list(self.locale_configs.keys())
    
    def get_locale_info(self, locale: str = None) -> LocaleConfig:
        """Get configuration information for a locale."""
        locale = locale or self.current_locale
        return self.locale_configs.get(locale, self.locale_configs[self.default_locale])
    
    def translate(self, 
                 message_id: str,
                 context: TranslationContext = None,
                 locale: str = None,
                 **kwargs) -> str:
        """Translate a message ID to the target locale."""
        
        locale = locale or self.current_locale
        context = context or TranslationContext()
        
        # Update context with kwargs
        if kwargs:
            context.variables.update(kwargs)
        
        # Create cache key
        cache_key = self._create_cache_key(message_id, context, locale)
        
        # Check cache first
        with self.cache_lock:
            if cache_key in self.translation_cache:
                self.translation_stats["cache_hits"] += 1
                return self.translation_cache[cache_key]
        
        # Perform translation
        translated = self._perform_translation(message_id, context, locale)
        
        # Cache result
        with self.cache_lock:
            self.translation_cache[cache_key] = translated
            self.translation_stats["cache_misses"] += 1
        
        return translated
    
    def _perform_translation(self, 
                           message_id: str,
                           context: TranslationContext,
                           locale: str) -> str:
        """Perform the actual translation with fallback logic."""
        
        # Try target locale first
        translation = self._get_translation_from_locale(message_id, context, locale)
        if translation:
            return self._interpolate_variables(translation, context.variables)
        
        # Try fallback locales
        locale_config = self.locale_configs.get(locale)
        if locale_config:
            for fallback_locale in locale_config.fallback_locales:
                translation = self._get_translation_from_locale(message_id, context, fallback_locale)
                if translation:
                    logger.debug(f"Using fallback locale {fallback_locale} for message {message_id}")
                    return self._interpolate_variables(translation, context.variables)
        
        # Last resort: return message ID with variables interpolated
        logger.warning(f"No translation found for message ID: {message_id} (locale: {locale})")
        return self._interpolate_variables(message_id, context.variables)
    
    def _get_translation_from_locale(self,
                                   message_id: str,
                                   context: TranslationContext,
                                   locale: str) -> Optional[str]:
        """Get translation from specific locale."""
        
        if locale not in self.translations:
            return None
        
        locale_translations = self.translations[locale]
        domain_translations = locale_translations.get(context.domain, {})
        
        # Direct lookup
        if message_id in domain_translations:
            translation = domain_translations[message_id]
            
            # Handle pluralization
            if context.pluralization and isinstance(translation, dict):
                plural_form = self._get_plural_form(context.pluralization, locale)
                return translation.get(plural_form, translation.get("other", message_id))
            
            return str(translation)
        
        # Context-specific lookup
        if context.context:
            contextual_key = f"{context.context}.{message_id}"
            if contextual_key in domain_translations:
                return str(domain_translations[contextual_key])
        
        return None
    
    def _get_plural_form(self, count: str, locale: str) -> str:
        """Determine plural form based on count and locale rules."""
        try:
            count_int = int(count)
        except ValueError:
            return "other"
        
        # Simplified plural rules - in real implementation would use CLDR data
        if locale in ["en"]:
            return "one" if count_int == 1 else "other"
        elif locale in ["fr", "es", "it", "pt"]:
            return "one" if count_int <= 1 else "other"
        elif locale in ["ru"]:
            # Russian has complex plural rules
            if count_int % 10 == 1 and count_int % 100 != 11:
                return "one"
            elif count_int % 10 in [2, 3, 4] and not (count_int % 100 in [12, 13, 14]):
                return "few"
            else:
                return "many"
        else:
            return "other"
    
    def _interpolate_variables(self, text: str, variables: Dict[str, Any]) -> str:
        """Interpolate variables into translated text."""
        if not variables:
            return text
        
        try:
            # Support both {variable} and %(variable)s formats
            if '{' in text:
                return text.format(**variables)
            elif '%(' in text:
                return text % variables
            else:
                return text
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Variable interpolation failed for text '{text}': {e}")
            return text
    
    def _create_cache_key(self, 
                         message_id: str,
                         context: TranslationContext,
                         locale: str) -> str:
        """Create cache key for translation."""
        key_parts = [
            locale,
            context.domain,
            message_id,
            context.context or "",
            context.pluralization or "",
            str(sorted(context.variables.items())) if context.variables else ""
        ]
        return "|".join(key_parts)
    
    def format_number(self, number: Union[int, float], locale: str = None) -> str:
        """Format number according to locale conventions."""
        locale = locale or self.current_locale
        locale_config = self.get_locale_info(locale)
        
        if BABEL_AVAILABLE:
            try:
                babel_locale = Locale.parse(locale.replace('-', '_'))
                return numbers.format_number(number, locale=babel_locale)
            except Exception as e:
                logger.debug(f"Babel number formatting failed: {e}")
        
        # Fallback formatting
        if isinstance(number, float):
            formatted = f"{number:,.{2}f}"
        else:
            formatted = f"{number:,}"
        
        # Apply locale-specific separators
        formatted = formatted.replace(',', '|||TEMP|||')  # Temporary replacement
        formatted = formatted.replace('.', locale_config.decimal_separator)
        formatted = formatted.replace('|||TEMP|||', locale_config.thousand_separator)
        
        return formatted
    
    def format_currency(self, 
                       amount: Union[int, float],
                       currency: str = None,
                       locale: str = None) -> str:
        """Format currency according to locale conventions."""
        locale = locale or self.current_locale
        locale_config = self.get_locale_info(locale)
        currency = currency or locale_config.currency_code
        
        if BABEL_AVAILABLE:
            try:
                babel_locale = Locale.parse(locale.replace('-', '_'))
                return numbers.format_currency(amount, currency, locale=babel_locale)
            except Exception as e:
                logger.debug(f"Babel currency formatting failed: {e}")
        
        # Fallback formatting
        formatted_number = self.format_number(amount, locale)
        
        # Simple currency formatting
        symbol = locale_config.currency_symbol
        if locale_config.text_direction == TextDirection.RTL:
            return f"{formatted_number} {symbol}"
        else:
            return f"{symbol}{formatted_number}"
    
    def format_datetime(self, 
                       dt: datetime.datetime,
                       format_type: str = "medium",
                       locale: str = None) -> str:
        """Format datetime according to locale conventions."""
        locale = locale or self.current_locale
        locale_config = self.get_locale_info(locale)
        
        if BABEL_AVAILABLE:
            try:
                babel_locale = Locale.parse(locale.replace('-', '_'))
                if format_type == "short":
                    return dates.format_datetime(dt, format='short', locale=babel_locale)
                elif format_type == "medium":
                    return dates.format_datetime(dt, format='medium', locale=babel_locale)
                elif format_type == "long":
                    return dates.format_datetime(dt, format='long', locale=babel_locale)
                else:
                    return dates.format_datetime(dt, format=format_type, locale=babel_locale)
            except Exception as e:
                logger.debug(f"Babel datetime formatting failed: {e}")
        
        # Fallback formatting
        if format_type == "date":
            return dt.strftime(locale_config.date_format)
        elif format_type == "time":
            return dt.strftime(locale_config.time_format)
        else:
            return dt.strftime(f"{locale_config.date_format} {locale_config.time_format}")
    
    def format_relative_time(self, 
                            dt: datetime.datetime,
                            locale: str = None) -> str:
        """Format relative time (e.g., '2 hours ago')."""
        locale = locale or self.current_locale
        
        if BABEL_AVAILABLE:
            try:
                babel_locale = Locale.parse(locale.replace('-', '_'))
                return dates.format_timedelta(
                    datetime.datetime.now() - dt,
                    locale=babel_locale,
                    add_direction=True
                )
            except Exception as e:
                logger.debug(f"Babel relative time formatting failed: {e}")
        
        # Fallback relative time formatting
        delta = datetime.datetime.now() - dt
        
        if delta.days > 0:
            return self.translate("relative_time.days_ago", count=str(delta.days), locale=locale)
        elif delta.seconds > 3600:
            hours = delta.seconds // 3600
            return self.translate("relative_time.hours_ago", count=str(hours), locale=locale)
        elif delta.seconds > 60:
            minutes = delta.seconds // 60
            return self.translate("relative_time.minutes_ago", count=str(minutes), locale=locale)
        else:
            return self.translate("relative_time.just_now", locale=locale)
    
    def get_translation_stats(self) -> Dict[str, Any]:
        """Get translation performance statistics."""
        total_requests = self.translation_stats["cache_hits"] + self.translation_stats["cache_misses"]
        cache_hit_rate = self.translation_stats["cache_hits"] / max(total_requests, 1)
        
        return {
            "total_translation_requests": total_requests,
            "cache_hits": self.translation_stats["cache_hits"],
            "cache_misses": self.translation_stats["cache_misses"],
            "cache_hit_rate": cache_hit_rate,
            "cached_translations": len(self.translation_cache),
            "supported_locales": len(self.locale_configs),
            "loaded_locales": len(self.translations)
        }
    
    def clear_cache(self):
        """Clear translation cache."""
        with self.cache_lock:
            self.translation_cache.clear()
        logger.info("Translation cache cleared")
    
    def reload_translations(self, locale: str = None):
        """Reload translations for specified locale or all locales."""
        if locale:
            self._load_locale_translations(locale)
        else:
            self._load_translations()
        
        self.clear_cache()
        logger.info(f"Translations reloaded for: {locale or 'all locales'}")


class MessageFormatter:
    """Advanced message formatting with locale-aware features."""
    
    def __init__(self, translation_manager: TranslationManager):
        self.translation_manager = translation_manager
    
    def format_benchmark_result(self, 
                               result_data: Dict[str, Any],
                               locale: str = None) -> Dict[str, str]:
        """Format benchmark results with proper localization."""
        locale = locale or self.translation_manager.current_locale
        
        formatted = {}
        
        # Format metrics
        if "metrics" in result_data:
            metrics = result_data["metrics"]
            formatted_metrics = {}
            
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    formatted_value = self.translation_manager.format_number(value, locale)
                    formatted_metrics[metric_name] = formatted_value
                else:
                    formatted_metrics[metric_name] = str(value)
            
            formatted["metrics"] = formatted_metrics
        
        # Format performance data
        if "performance" in result_data:
            performance = result_data["performance"]
            formatted_performance = {}
            
            for perf_name, value in performance.items():
                if isinstance(value, (int, float)):
                    if "latency" in perf_name.lower() or "time" in perf_name.lower():
                        # Format as duration
                        formatted_value = self.format_duration(value, locale)
                    elif "memory" in perf_name.lower() or "vram" in perf_name.lower():
                        # Format as data size
                        formatted_value = self.format_data_size(value, locale)
                    else:
                        formatted_value = self.translation_manager.format_number(value, locale)
                    
                    formatted_performance[perf_name] = formatted_value
                else:
                    formatted_performance[perf_name] = str(value)
            
            formatted["performance"] = formatted_performance
        
        return formatted
    
    def format_duration(self, milliseconds: float, locale: str = None) -> str:
        """Format duration in milliseconds to human-readable format."""
        locale = locale or self.translation_manager.current_locale
        
        if milliseconds < 1000:
            return self.translation_manager.translate(
                "duration.milliseconds",
                TranslationContext(domain="formatting"),
                locale=locale,
                value=int(milliseconds)
            )
        elif milliseconds < 60000:
            seconds = milliseconds / 1000
            return self.translation_manager.translate(
                "duration.seconds",
                TranslationContext(domain="formatting"),
                locale=locale,
                value=f"{seconds:.1f}"
            )
        else:
            minutes = milliseconds / 60000
            return self.translation_manager.translate(
                "duration.minutes",
                TranslationContext(domain="formatting"),
                locale=locale,
                value=f"{minutes:.1f}"
            )
    
    def format_data_size(self, bytes_value: float, locale: str = None) -> str:
        """Format data size in bytes to human-readable format."""
        locale = locale or self.translation_manager.current_locale
        
        units = ["B", "KB", "MB", "GB", "TB"]
        size = float(bytes_value)
        unit_index = 0
        
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
        
        formatted_size = self.translation_manager.format_number(size, locale)
        unit = units[unit_index]
        
        return self.translation_manager.translate(
            f"data_size.{unit.lower()}",
            TranslationContext(domain="formatting"),
            locale=locale,
            value=formatted_size
        )
    
    def format_error_message(self, 
                           error_code: str,
                           error_details: Dict[str, Any] = None,
                           locale: str = None) -> str:
        """Format error message with localization."""
        locale = locale or self.translation_manager.current_locale
        error_details = error_details or {}
        
        return self.translation_manager.translate(
            f"errors.{error_code}",
            TranslationContext(domain="errors", variables=error_details),
            locale=locale
        )
    
    def format_validation_message(self,
                                field_name: str,
                                validation_type: str,
                                locale: str = None,
                                **kwargs) -> str:
        """Format field validation message."""
        locale = locale or self.translation_manager.current_locale
        
        return self.translation_manager.translate(
            f"validation.{validation_type}",
            TranslationContext(domain="validation", variables={"field": field_name, **kwargs}),
            locale=locale
        )


class LocaleDetector:
    """Automatic locale detection from various sources."""
    
    @staticmethod
    def detect_from_http_headers(accept_language: str) -> List[Tuple[str, float]]:
        """Parse Accept-Language header and return ordered list of locales with quality scores."""
        if not accept_language:
            return [("en", 1.0)]
        
        locales = []
        
        # Parse Accept-Language header
        # Format: en-US,en;q=0.9,es;q=0.8,fr;q=0.7
        for lang_range in accept_language.split(','):
            lang_range = lang_range.strip()
            
            if ';q=' in lang_range:
                lang, quality_str = lang_range.split(';q=', 1)
                try:
                    quality = float(quality_str)
                except ValueError:
                    quality = 1.0
            else:
                lang = lang_range
                quality = 1.0
            
            # Normalize language code
            lang = lang.strip().lower()
            if '-' in lang:
                # Convert en-US to en-us, then to en-US format
                parts = lang.split('-')
                if len(parts) == 2:
                    lang = f"{parts[0]}-{parts[1].upper()}"
            
            locales.append((lang, quality))
        
        # Sort by quality score (descending)
        locales.sort(key=lambda x: x[1], reverse=True)
        
        return locales
    
    @staticmethod
    def detect_from_environment() -> Optional[str]:
        """Detect locale from environment variables."""
        env_vars = ['LANG', 'LC_ALL', 'LC_CTYPE', 'LANGUAGE']
        
        for var in env_vars:
            value = os.environ.get(var)
            if value:
                # Extract locale from environment variable
                # Format might be: en_US.UTF-8, en_US, en
                locale_match = re.match(r'^([a-z]{2}(_[A-Z]{2})?)', value)
                if locale_match:
                    locale = locale_match.group(1).replace('_', '-')
                    return locale
        
        return None
    
    @staticmethod
    def detect_from_timezone() -> Optional[str]:
        """Infer likely locale from system timezone."""
        try:
            import time
            timezone = time.tzname[0] if hasattr(time, 'tzname') else None
            
            # Simple mapping of timezones to likely locales
            timezone_locale_map = {
                'EST': 'en-US',
                'PST': 'en-US', 
                'GMT': 'en-GB',
                'CET': 'de-DE',
                'JST': 'ja-JP',
                'CST': 'zh-CN',
                'IST': 'hi-IN'
            }
            
            return timezone_locale_map.get(timezone)
            
        except Exception:
            return None
    
    @classmethod
    def detect_best_locale(cls,
                         supported_locales: List[str],
                         accept_language: str = None) -> str:
        """Detect the best locale from available sources."""
        
        # Try HTTP Accept-Language header first
        if accept_language:
            preferred_locales = cls.detect_from_http_headers(accept_language)
            for locale, _ in preferred_locales:
                if locale in supported_locales:
                    return locale
                
                # Try language-only match (e.g., 'en' for 'en-US')
                lang_only = locale.split('-')[0]
                for supported in supported_locales:
                    if supported.startswith(lang_only):
                        return supported
        
        # Try environment variables
        env_locale = cls.detect_from_environment()
        if env_locale and env_locale in supported_locales:
            return env_locale
        
        # Try timezone inference
        tz_locale = cls.detect_from_timezone()
        if tz_locale and tz_locale in supported_locales:
            return tz_locale
        
        # Default fallback
        return supported_locales[0] if supported_locales else "en"


# Global translation manager instance
_translation_manager = None

def get_translation_manager() -> TranslationManager:
    """Get global translation manager instance."""
    global _translation_manager
    if _translation_manager is None:
        _translation_manager = TranslationManager()
    return _translation_manager


def translate(message_id: str, **kwargs) -> str:
    """Convenience function for translation."""
    return get_translation_manager().translate(message_id, **kwargs)


def set_locale(locale: str) -> bool:
    """Convenience function to set locale."""
    return get_translation_manager().set_locale(locale)


def format_number(number: Union[int, float], locale: str = None) -> str:
    """Convenience function for number formatting."""
    return get_translation_manager().format_number(number, locale)


def format_currency(amount: Union[int, float], currency: str = None, locale: str = None) -> str:
    """Convenience function for currency formatting."""
    return get_translation_manager().format_currency(amount, currency, locale)


def format_datetime(dt: datetime.datetime, format_type: str = "medium", locale: str = None) -> str:
    """Convenience function for datetime formatting."""
    return get_translation_manager().format_datetime(dt, format_type, locale)


def create_sample_translations():
    """Create sample translation files for development."""
    base_path = Path(__file__).parent / "locales"
    
    # Sample translations for different domains
    sample_translations = {
        "en": {
            "benchmark": {
                "model_evaluation": "Model Evaluation",
                "benchmark_complete": "Benchmark completed successfully",
                "processing_prompts": "Processing {count} prompts",
                "video_generation_failed": "Video generation failed for prompt: {prompt}"
            },
            "errors": {
                "model_not_found": "Model '{model_name}' not found",
                "insufficient_memory": "Insufficient GPU memory: {required_gb}GB required, {available_gb}GB available",
                "invalid_parameters": "Invalid parameters: {details}"
            },
            "formatting": {
                "duration.milliseconds": "{value}ms",
                "duration.seconds": "{value}s", 
                "duration.minutes": "{value}min",
                "data_size.b": "{value} B",
                "data_size.kb": "{value} KB",
                "data_size.mb": "{value} MB",
                "data_size.gb": "{value} GB"
            }
        },
        "es": {
            "benchmark": {
                "model_evaluation": "Evaluación del Modelo",
                "benchmark_complete": "Benchmark completado exitosamente",
                "processing_prompts": "Procesando {count} prompts",
                "video_generation_failed": "La generación de video falló para el prompt: {prompt}"
            },
            "errors": {
                "model_not_found": "Modelo '{model_name}' no encontrado",
                "insufficient_memory": "Memoria GPU insuficiente: {required_gb}GB requeridos, {available_gb}GB disponibles",
                "invalid_parameters": "Parámetros inválidos: {details}"
            }
        },
        "zh-CN": {
            "benchmark": {
                "model_evaluation": "模型评估",
                "benchmark_complete": "基准测试成功完成",
                "processing_prompts": "正在处理 {count} 个提示",
                "video_generation_failed": "提示视频生成失败：{prompt}"
            },
            "errors": {
                "model_not_found": "未找到模型 '{model_name}'",
                "insufficient_memory": "GPU内存不足：需要{required_gb}GB，可用{available_gb}GB",
                "invalid_parameters": "无效参数：{details}"
            }
        }
    }
    
    # Create translation files
    for locale, domains in sample_translations.items():
        locale_dir = base_path / locale
        locale_dir.mkdir(parents=True, exist_ok=True)
        
        for domain, translations in domains.items():
            domain_file = locale_dir / f"{domain}.json"
            with open(domain_file, 'w', encoding='utf-8') as f:
                json.dump(translations, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created sample translations for {locale}")


if __name__ == "__main__":
    # Create sample translations for development
    create_sample_translations()
    
    # Test the translation system
    tm = TranslationManager()
    
    # Test basic translation
    print("English:", tm.translate("benchmark.model_evaluation"))
    
    # Test with variables
    print("English with variables:", tm.translate(
        "benchmark.processing_prompts", 
        count="5"
    ))
    
    # Test locale switching
    tm.set_locale("es")
    print("Spanish:", tm.translate("benchmark.model_evaluation"))
    
    # Test number formatting
    tm.set_locale("en")
    print("English number:", tm.format_number(1234567.89))
    
    tm.set_locale("es")
    print("Spanish number:", tm.format_number(1234567.89))
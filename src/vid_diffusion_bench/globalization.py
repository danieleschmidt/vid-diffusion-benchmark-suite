"""Global-first implementation with multi-region support and internationalization.

This module provides comprehensive globalization features including:
- Multi-language support (i18n) for 6+ languages
- Multi-region deployment capabilities
- Compliance with GDPR, CCPA, PDPA
- Cross-platform compatibility
- Cultural adaptation for different markets
"""

import logging
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
import locale
import gettext
from functools import lru_cache
import re

logger = logging.getLogger(__name__)


@dataclass 
class RegionConfig:
    """Configuration for a specific region."""
    region_code: str  # e.g., 'us-east-1', 'eu-west-1'
    country_codes: List[str]  # ISO country codes
    languages: List[str]  # Language codes
    data_residency: str  # Data residency requirements
    compliance_frameworks: List[str]  # GDPR, CCPA, etc.
    currency: str  # ISO currency code
    timezone: str  # Standard timezone
    cultural_preferences: Dict[str, Any]
    deployment_endpoints: Dict[str, str]
    
    @property
    def primary_language(self) -> str:
        """Get primary language for the region."""
        return self.languages[0] if self.languages else 'en'


@dataclass
class LocalizationContext:
    """Context for localization decisions."""
    user_language: str = 'en'
    user_region: str = 'global'
    user_country: str = 'US'
    user_timezone: str = 'UTC'
    currency: str = 'USD'
    date_format: str = '%Y-%m-%d'
    number_format: str = 'en_US'
    cultural_context: Dict[str, Any] = field(default_factory=dict)


class TranslationManager:
    """Manages translations and localization."""
    
    def __init__(self, translations_dir: str = None):
        """Initialize translation manager.
        
        Args:
            translations_dir: Directory containing translation files
        """
        self.translations_dir = Path(translations_dir or Path(__file__).parent / 'translations')
        self.translations_dir.mkdir(exist_ok=True)
        
        # Supported languages with full names
        self.supported_languages = {
            'en': 'English',
            'es': 'Español',
            'fr': 'Français', 
            'de': 'Deutsch',
            'ja': '日本語',
            'zh': '中文',
            'pt': 'Português',
            'it': 'Italiano',
            'ru': 'Русский'
        }
        
        # Load translations
        self._translations: Dict[str, Dict[str, str]] = {}
        self._load_translations()
        
        # Set up gettext
        self._setup_gettext()
        
    def _load_translations(self):
        """Load translation files."""
        # Default English strings
        self._translations['en'] = {
            'benchmark.start': 'Starting benchmark for model {model_name}',
            'benchmark.complete': 'Benchmark completed successfully',
            'benchmark.failed': 'Benchmark failed with error: {error}',
            'metrics.fvd': 'Fréchet Video Distance',
            'metrics.is': 'Inception Score',
            'metrics.clip': 'CLIP Similarity',
            'metrics.temporal': 'Temporal Consistency',
            'validation.prompt_empty': 'Prompt cannot be empty',
            'validation.invalid_params': 'Invalid parameters: {params}',
            'error.gpu_memory': 'Insufficient GPU memory',
            'error.model_load': 'Failed to load model: {model_name}',
            'status.processing': 'Processing...',
            'status.ready': 'Ready',
            'config.batch_size': 'Batch Size',
            'config.num_frames': 'Number of Frames',
            'config.resolution': 'Resolution',
            'report.summary': 'Benchmark Summary',
            'report.model_ranking': 'Model Rankings',
            'report.performance': 'Performance Metrics',
            'export.csv': 'Export to CSV',
            'export.json': 'Export to JSON',
            'export.pdf': 'Export to PDF'
        }
        
        # Load other language files
        for lang_code in self.supported_languages:
            if lang_code != 'en':
                self._load_language_file(lang_code)
    
    def _load_language_file(self, lang_code: str):
        """Load translations for a specific language."""
        lang_file = self.translations_dir / f'{lang_code}.json'
        
        if lang_file.exists():
            try:
                with open(lang_file, 'r', encoding='utf-8') as f:
                    self._translations[lang_code] = json.load(f)
                logger.info(f"Loaded translations for {lang_code}")
            except Exception as e:
                logger.error(f"Failed to load translations for {lang_code}: {e}")
                self._translations[lang_code] = {}
        else:
            # Create template file with English strings for translation
            self._create_translation_template(lang_code)
            self._translations[lang_code] = self._translations['en'].copy()
    
    def _create_translation_template(self, lang_code: str):
        """Create translation template file."""
        lang_file = self.translations_dir / f'{lang_code}.json'
        
        # Add translation notes for context
        template = {key: f"TODO: Translate '{value}'" for key, value in self._translations['en'].items()}
        
        try:
            with open(lang_file, 'w', encoding='utf-8') as f:
                json.dump(template, f, ensure_ascii=False, indent=2)
            logger.info(f"Created translation template for {lang_code}")
        except Exception as e:
            logger.error(f"Failed to create translation template for {lang_code}: {e}")
    
    def _setup_gettext(self):
        """Set up gettext for professional translations."""
        self._gettext_translations = {}
        
        for lang_code in self.supported_languages:
            locale_dir = self.translations_dir / 'locale'
            mo_file = locale_dir / lang_code / 'LC_MESSAGES' / 'messages.mo'
            
            if mo_file.exists():
                try:
                    translation = gettext.translation(
                        'messages', 
                        localedir=str(locale_dir),
                        languages=[lang_code]
                    )
                    self._gettext_translations[lang_code] = translation
                except Exception as e:
                    logger.warning(f"Failed to load gettext for {lang_code}: {e}")
    
    def translate(self, key: str, lang: str = 'en', **kwargs) -> str:
        """Translate a string key to the specified language.
        
        Args:
            key: Translation key
            lang: Target language code
            **kwargs: Variables for string formatting
            
        Returns:
            Translated and formatted string
        """
        # Use gettext if available
        if lang in self._gettext_translations:
            try:
                translation = self._gettext_translations[lang]
                translated = translation.gettext(key)
                if translated != key:  # Found translation
                    return translated.format(**kwargs) if kwargs else translated
            except Exception as e:
                logger.warning(f"Gettext translation failed for {key} ({lang}): {e}")
        
        # Fallback to JSON translations
        if lang in self._translations and key in self._translations[lang]:
            template = self._translations[lang][key]
        elif key in self._translations['en']:
            template = self._translations['en'][key]
            logger.debug(f"Falling back to English for key '{key}' in language '{lang}'")
        else:
            logger.warning(f"Translation key '{key}' not found")
            return key
        
        # Format with variables
        try:
            return template.format(**kwargs) if kwargs else template
        except KeyError as e:
            logger.error(f"Missing variable for translation '{key}': {e}")
            return template
    
    def get_available_languages(self) -> Dict[str, str]:
        """Get list of available languages with their native names."""
        return self.supported_languages.copy()
    
    def add_translation(self, key: str, lang: str, value: str):
        """Add or update a translation."""
        if lang not in self._translations:
            self._translations[lang] = {}
        
        self._translations[lang][key] = value
        
        # Save to file
        self._save_language_file(lang)
    
    def _save_language_file(self, lang_code: str):
        """Save translations to file."""
        lang_file = self.translations_dir / f'{lang_code}.json'
        
        try:
            with open(lang_file, 'w', encoding='utf-8') as f:
                json.dump(self._translations[lang_code], f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save translations for {lang_code}: {e}")


class RegionManager:
    """Manages regional configurations and deployments."""
    
    def __init__(self):
        """Initialize region manager."""
        self.regions = self._initialize_regions()
        self.current_region = 'global'
        
    def _initialize_regions(self) -> Dict[str, RegionConfig]:
        """Initialize regional configurations."""
        return {
            'us-east-1': RegionConfig(
                region_code='us-east-1',
                country_codes=['US', 'CA'],
                languages=['en', 'es'],
                data_residency='US',
                compliance_frameworks=['SOC2', 'HIPAA'],
                currency='USD',
                timezone='America/New_York',
                cultural_preferences={
                    'date_format': 'MM/DD/YYYY',
                    'measurement_system': 'imperial',
                    'currency_position': 'before'
                },
                deployment_endpoints={
                    'api': 'https://api-us-east.vid-bench.com',
                    'web': 'https://us.vid-bench.com'
                }
            ),
            'eu-west-1': RegionConfig(
                region_code='eu-west-1', 
                country_codes=['DE', 'FR', 'IT', 'ES', 'NL', 'BE'],
                languages=['en', 'de', 'fr', 'it', 'es'],
                data_residency='EU',
                compliance_frameworks=['GDPR'],
                currency='EUR',
                timezone='Europe/Berlin',
                cultural_preferences={
                    'date_format': 'DD/MM/YYYY',
                    'measurement_system': 'metric',
                    'currency_position': 'after'
                },
                deployment_endpoints={
                    'api': 'https://api-eu-west.vid-bench.com',
                    'web': 'https://eu.vid-bench.com'
                }
            ),
            'ap-southeast-1': RegionConfig(
                region_code='ap-southeast-1',
                country_codes=['SG', 'MY', 'TH', 'ID'],
                languages=['en', 'zh'],
                data_residency='Singapore', 
                compliance_frameworks=['PDPA', 'GDPR'],
                currency='SGD',
                timezone='Asia/Singapore',
                cultural_preferences={
                    'date_format': 'DD/MM/YYYY',
                    'measurement_system': 'metric',
                    'currency_position': 'before'
                },
                deployment_endpoints={
                    'api': 'https://api-ap-southeast.vid-bench.com',
                    'web': 'https://ap.vid-bench.com'
                }
            ),
            'ap-northeast-1': RegionConfig(
                region_code='ap-northeast-1',
                country_codes=['JP'],
                languages=['ja', 'en'],
                data_residency='Japan',
                compliance_frameworks=['Act on Protection of Personal Information'],
                currency='JPY',
                timezone='Asia/Tokyo',
                cultural_preferences={
                    'date_format': 'YYYY/MM/DD',
                    'measurement_system': 'metric',
                    'currency_position': 'after'
                },
                deployment_endpoints={
                    'api': 'https://api-ap-northeast.vid-bench.com',
                    'web': 'https://jp.vid-bench.com'
                }
            )
        }
    
    def get_region_for_country(self, country_code: str) -> Optional[str]:
        """Get the appropriate region for a country."""
        for region_code, region in self.regions.items():
            if country_code.upper() in region.country_codes:
                return region_code
        return None
    
    def get_compliance_requirements(self, region_code: str) -> List[str]:
        """Get compliance requirements for a region."""
        if region_code in self.regions:
            return self.regions[region_code].compliance_frameworks
        return []
    
    def is_gdpr_region(self, region_code: str) -> bool:
        """Check if region requires GDPR compliance."""
        requirements = self.get_compliance_requirements(region_code)
        return 'GDPR' in requirements


class ComplianceManager:
    """Manages compliance with various regulatory frameworks."""
    
    def __init__(self):
        """Initialize compliance manager."""
        self.compliance_configs = {
            'GDPR': {
                'data_retention_days': 1095,  # 3 years
                'consent_required': True,
                'right_to_delete': True,
                'data_portability': True,
                'privacy_by_design': True,
                'requires_dpo': False,  # For small organizations
                'cookie_consent': True
            },
            'CCPA': {
                'data_retention_days': 1095,
                'consent_required': False,  # Opt-out model
                'right_to_delete': True,
                'data_portability': True,
                'privacy_by_design': False,
                'sale_opt_out': True
            },
            'PDPA': {
                'data_retention_days': 1095,
                'consent_required': True,
                'right_to_delete': True,
                'data_portability': True,
                'privacy_by_design': True,
                'notification_required': True
            },
            'HIPAA': {
                'data_retention_days': 2555,  # 7 years
                'encryption_required': True,
                'access_logging': True,
                'breach_notification': True,
                'business_associate_agreement': True
            }
        }
    
    def get_compliance_requirements(self, framework: str) -> Dict[str, Any]:
        """Get requirements for a compliance framework."""
        return self.compliance_configs.get(framework, {})
    
    def validate_data_retention(self, 
                               data_age_days: int,
                               frameworks: List[str]) -> Dict[str, bool]:
        """Validate data retention against compliance frameworks."""
        results = {}
        
        for framework in frameworks:
            config = self.compliance_configs.get(framework, {})
            retention_limit = config.get('data_retention_days', float('inf'))
            results[framework] = data_age_days <= retention_limit
        
        return results
    
    def get_privacy_controls(self, frameworks: List[str]) -> Dict[str, Any]:
        """Get required privacy controls for frameworks."""
        controls = {
            'consent_required': False,
            'right_to_delete': False,
            'data_portability': False,
            'privacy_by_design': False,
            'encryption_required': False,
            'access_logging': False
        }
        
        for framework in frameworks:
            config = self.compliance_configs.get(framework, {})
            for control, required in config.items():
                if control in controls:
                    controls[control] = controls[control] or required
        
        return controls


class CulturalAdapter:
    """Adapts content and behavior for different cultural contexts."""
    
    def __init__(self):
        """Initialize cultural adapter."""
        self.cultural_patterns = {
            'date_formats': {
                'US': '%m/%d/%Y',
                'EU': '%d/%m/%Y', 
                'ISO': '%Y-%m-%d',
                'JP': '%Y/%m/%d'
            },
            'number_formats': {
                'US': {'decimal': '.', 'thousands': ','},
                'EU': {'decimal': ',', 'thousands': '.'},
                'IN': {'decimal': '.', 'thousands': ',', 'lakh_crore': True}
            },
            'currency_formats': {
                'USD': {'symbol': '$', 'position': 'before'},
                'EUR': {'symbol': '€', 'position': 'after'},
                'JPY': {'symbol': '¥', 'position': 'before'},
                'GBP': {'symbol': '£', 'position': 'before'}
            },
            'measurement_systems': {
                'US': 'imperial',
                'UK': 'mixed',
                'default': 'metric'
            }
        }
    
    def format_date(self, date: datetime, context: LocalizationContext) -> str:
        """Format date according to cultural preferences."""
        format_pattern = self.cultural_patterns['date_formats'].get(
            context.user_country, 
            context.date_format
        )
        
        try:
            return date.strftime(format_pattern)
        except Exception:
            return date.strftime('%Y-%m-%d')  # ISO fallback
    
    def format_number(self, number: float, context: LocalizationContext) -> str:
        """Format number according to cultural preferences."""
        number_config = self.cultural_patterns['number_formats'].get(
            context.user_country,
            self.cultural_patterns['number_formats']['US']
        )
        
        try:
            # Handle Indian lakh/crore system
            if number_config.get('lakh_crore') and number >= 100000:
                if number >= 10000000:  # 1 crore
                    crores = number / 10000000
                    return f"{crores:.2f} crore"
                else:  # 1 lakh
                    lakhs = number / 100000
                    return f"{lakhs:.2f} lakh"
            
            # Standard formatting
            decimal_sep = number_config['decimal']
            thousands_sep = number_config['thousands']
            
            # Format the number
            formatted = f"{number:,.2f}"
            
            # Replace separators
            if decimal_sep != '.' or thousands_sep != ',':
                parts = formatted.split('.')
                integer_part = parts[0].replace(',', thousands_sep)
                if len(parts) > 1:
                    formatted = f"{integer_part}{decimal_sep}{parts[1]}"
                else:
                    formatted = integer_part
            
            return formatted
            
        except Exception as e:
            logger.error(f"Number formatting error: {e}")
            return str(number)
    
    def format_currency(self, amount: float, context: LocalizationContext) -> str:
        """Format currency according to cultural preferences."""
        currency_config = self.cultural_patterns['currency_formats'].get(
            context.currency,
            {'symbol': context.currency, 'position': 'before'}
        )
        
        formatted_amount = self.format_number(amount, context)
        symbol = currency_config['symbol']
        
        if currency_config['position'] == 'before':
            return f"{symbol}{formatted_amount}"
        else:
            return f"{formatted_amount} {symbol}"
    
    def adapt_content_length(self, text: str, language: str) -> str:
        """Adapt content length for different languages."""
        # Some languages are more verbose than others
        expansion_factors = {
            'de': 1.3,  # German tends to be longer
            'fr': 1.2,  # French slightly longer
            'ja': 0.7,  # Japanese more compact
            'zh': 0.6,  # Chinese very compact
            'es': 1.15, # Spanish slightly longer
            'en': 1.0   # English baseline
        }
        
        factor = expansion_factors.get(language, 1.0)
        
        if factor > 1.1:  # Language tends to be verbose
            # Could implement text compression strategies
            return text
        elif factor < 0.9:  # Language tends to be compact
            # Could implement text expansion strategies
            return text
        
        return text


class GlobalizationManager:
    """Main manager for all globalization features."""
    
    def __init__(self, default_language: str = 'en', default_region: str = 'global'):
        """Initialize globalization manager.
        
        Args:
            default_language: Default language code
            default_region: Default region code
        """
        self.translation_manager = TranslationManager()
        self.region_manager = RegionManager()
        self.compliance_manager = ComplianceManager()
        self.cultural_adapter = CulturalAdapter()
        
        self.default_context = LocalizationContext(
            user_language=default_language,
            user_region=default_region
        )
        
        logger.info(f"GlobalizationManager initialized with language={default_language}, region={default_region}")
    
    def create_context(self, 
                      language: str = None,
                      region: str = None,
                      country: str = None,
                      **kwargs) -> LocalizationContext:
        """Create localization context."""
        # Auto-detect region from country if not provided
        if country and not region:
            region = self.region_manager.get_region_for_country(country)
        
        # Get region config for defaults
        region_config = None
        if region and region in self.region_manager.regions:
            region_config = self.region_manager.regions[region]
            
        context = LocalizationContext(
            user_language=language or (region_config.primary_language if region_config else self.default_context.user_language),
            user_region=region or self.default_context.user_region,
            user_country=country or 'US',
            currency=region_config.currency if region_config else 'USD',
            **kwargs
        )
        
        return context
    
    def localize_message(self, 
                        key: str,
                        context: LocalizationContext = None,
                        **kwargs) -> str:
        """Localize a message with full cultural adaptation."""
        context = context or self.default_context
        
        # Get base translation
        message = self.translation_manager.translate(
            key, 
            context.user_language,
            **kwargs
        )
        
        # Apply cultural adaptation
        message = self.cultural_adapter.adapt_content_length(
            message, 
            context.user_language
        )
        
        return message
    
    def localize_data(self, 
                     data: Dict[str, Any],
                     context: LocalizationContext = None) -> Dict[str, Any]:
        """Localize data structures with dates, numbers, currencies."""
        context = context or self.default_context
        localized_data = data.copy()
        
        for key, value in localized_data.items():
            if isinstance(value, datetime):
                localized_data[key] = self.cultural_adapter.format_date(value, context)
            elif isinstance(value, (int, float)) and 'currency' in key.lower():
                localized_data[key] = self.cultural_adapter.format_currency(value, context)
            elif isinstance(value, (int, float)) and any(keyword in key.lower() for keyword in ['price', 'cost', 'amount']):
                localized_data[key] = self.cultural_adapter.format_currency(value, context) 
            elif isinstance(value, float):
                localized_data[key] = self.cultural_adapter.format_number(value, context)
            elif isinstance(value, dict):
                localized_data[key] = self.localize_data(value, context)
        
        return localized_data
    
    def get_compliance_info(self, region: str) -> Dict[str, Any]:
        """Get compliance information for a region."""
        frameworks = self.region_manager.get_compliance_requirements(region)
        
        compliance_info = {
            'frameworks': frameworks,
            'controls': self.compliance_manager.get_privacy_controls(frameworks),
            'is_gdpr_region': self.region_manager.is_gdpr_region(region)
        }
        
        return compliance_info
    
    def validate_data_handling(self, 
                              region: str,
                              data_age_days: int) -> Dict[str, Any]:
        """Validate data handling compliance."""
        frameworks = self.region_manager.get_compliance_requirements(region)
        retention_validation = self.compliance_manager.validate_data_retention(
            data_age_days, 
            frameworks
        )
        
        return {
            'compliant': all(retention_validation.values()),
            'frameworks': retention_validation,
            'required_actions': self._get_required_actions(retention_validation)
        }
    
    def _get_required_actions(self, validation_results: Dict[str, bool]) -> List[str]:
        """Get required actions for compliance."""
        actions = []
        
        for framework, is_compliant in validation_results.items():
            if not is_compliant:
                actions.append(f"Data retention exceeded for {framework} - consider deletion or archival")
        
        return actions
    
    def generate_localization_report(self) -> Dict[str, Any]:
        """Generate comprehensive localization status report."""
        return {
            'supported_languages': self.translation_manager.get_available_languages(),
            'supported_regions': list(self.region_manager.regions.keys()),
            'compliance_frameworks': list(self.compliance_manager.compliance_configs.keys()),
            'translation_coverage': {
                lang: len(self.translation_manager._translations.get(lang, {}))
                for lang in self.translation_manager.supported_languages
            },
            'cultural_adaptations': {
                'date_formats': len(self.cultural_adapter.cultural_patterns['date_formats']),
                'number_formats': len(self.cultural_adapter.cultural_patterns['number_formats']),
                'currency_formats': len(self.cultural_adapter.cultural_patterns['currency_formats'])
            }
        }
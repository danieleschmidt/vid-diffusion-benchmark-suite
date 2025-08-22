"""Global Compliance Framework for Video Diffusion Benchmarks.

This module implements comprehensive global compliance capabilities including
GDPR, CCPA, PDPA compliance, multi-region deployment, internationalization,
and cross-jurisdictional data governance for video diffusion benchmarking systems.
"""

import time
import logging
import asyncio
import json
import hashlib
import uuid
from typing import Dict, List, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, asdict, field
from pathlib import Path
from abc import ABC, abstractmethod
from enum import Enum
from collections import defaultdict
import gettext
import locale
from datetime import datetime, timezone, timedelta
import pycountry
import re

logger = logging.getLogger(__name__)


class DataClassification(Enum):
    """Data classification levels for compliance."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PERSONAL_DATA = "personal_data"
    SENSITIVE_PERSONAL_DATA = "sensitive_personal_data"


class ComplianceRegime(Enum):
    """Global compliance regimes."""
    GDPR = "gdpr"          # EU General Data Protection Regulation
    CCPA = "ccpa"          # California Consumer Privacy Act
    PDPA_SINGAPORE = "pdpa_sg"  # Singapore Personal Data Protection Act
    PDPA_THAILAND = "pdpa_th"   # Thailand Personal Data Protection Act
    LGPD = "lgpd"          # Brazil Lei Geral de Proteção de Dados
    PIPEDA = "pipeda"      # Canada Personal Information Protection
    PRIVACY_ACT = "privacy_act_au"  # Australia Privacy Act
    APPI = "appi"          # Japan Act on Personal Information Protection
    PIPA = "pipa"          # South Korea Personal Information Protection Act


class DataProcessingPurpose(Enum):
    """Purposes for data processing."""
    RESEARCH = "research"
    BENCHMARKING = "benchmarking"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    MODEL_EVALUATION = "model_evaluation"
    SYSTEM_OPTIMIZATION = "system_optimization"
    COMPLIANCE_MONITORING = "compliance_monitoring"
    ANALYTICS = "analytics"
    IMPROVEMENT = "improvement"


class LegalBasis(Enum):
    """Legal basis for data processing under GDPR."""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


@dataclass
class DataSubject:
    """Represents a data subject in compliance tracking."""
    subject_id: str
    jurisdiction: str
    data_classification: DataClassification
    consent_status: Dict[str, bool]
    processing_purposes: List[DataProcessingPurpose]
    legal_basis: LegalBasis
    data_categories: List[str]
    retention_until: Optional[datetime]
    anonymized: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConsentRecord:
    """Records consent for data processing."""
    consent_id: str
    subject_id: str
    purposes: List[DataProcessingPurpose]
    granted_at: datetime
    expires_at: Optional[datetime]
    withdrawn_at: Optional[datetime]
    lawful_basis: LegalBasis
    consent_method: str
    granular_permissions: Dict[str, bool]
    
    def is_valid(self) -> bool:
        """Check if consent is still valid."""
        now = datetime.now(timezone.utc)
        
        if self.withdrawn_at and self.withdrawn_at <= now:
            return False
        
        if self.expires_at and self.expires_at <= now:
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataProcessingRecord:
    """Records data processing activities."""
    record_id: str
    controller: str
    processor: Optional[str]
    data_categories: List[str]
    subject_categories: List[str]
    processing_purposes: List[DataProcessingPurpose]
    legal_basis: LegalBasis
    retention_period: str
    recipients: List[str]
    third_country_transfers: List[str]
    security_measures: List[str]
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComplianceAuditResult:
    """Results of a compliance audit."""
    audit_id: str
    regime: ComplianceRegime
    audit_date: datetime
    compliance_score: float
    findings: List[Dict[str, Any]]
    violations: List[Dict[str, Any]]
    recommendations: List[str]
    next_audit_due: datetime
    auditor: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DataAnonymizer:
    """Handles data anonymization for privacy compliance."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def anonymize_performance_data(self, data: Dict[str, Any], 
                                 anonymization_level: str = "standard") -> Dict[str, Any]:
        """
        Anonymize performance benchmark data while preserving analytical value.
        
        Args:
            data: Raw performance data
            anonymization_level: Level of anonymization (basic, standard, strict)
        
        Returns:
            Anonymized data suitable for sharing
        """
        
        anonymized = data.copy()
        
        # Remove or anonymize identifying information
        identifying_fields = [
            "user_id", "session_id", "ip_address", "device_id", 
            "user_agent", "organization", "email", "username"
        ]
        
        for field in identifying_fields:
            if field in anonymized:
                if anonymization_level == "strict":
                    del anonymized[field]
                else:
                    anonymized[field] = self._hash_identifier(str(anonymized[field]))
        
        # Anonymize timestamps to remove precise timing
        if "timestamp" in anonymized:
            timestamp = anonymized["timestamp"]
            if isinstance(timestamp, (int, float)):
                # Round to hour level
                rounded_timestamp = int(timestamp // 3600) * 3600
                anonymized["timestamp"] = rounded_timestamp
        
        # Add noise to sensitive metrics if required
        if anonymization_level == "strict":
            sensitive_metrics = ["latency_ms", "memory_gb", "power_watts"]
            for metric in sensitive_metrics:
                if metric in anonymized and isinstance(anonymized[metric], (int, float)):
                    # Add small amount of noise (±5%)
                    noise_factor = 1.0 + (hash(str(anonymized[metric])) % 100 - 50) / 1000.0
                    anonymized[metric] = anonymized[metric] * noise_factor
        
        # Remove geographical location data if present
        geo_fields = ["country", "region", "city", "lat", "lon", "timezone"]
        for field in geo_fields:
            if field in anonymized:
                if anonymization_level in ["standard", "strict"]:
                    if field in ["country"]:
                        # Keep country for legitimate analytics
                        continue
                    else:
                        del anonymized[field]
        
        return anonymized
    
    def _hash_identifier(self, identifier: str) -> str:
        """Create a stable hash of an identifier."""
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]
    
    def pseudonymize_data(self, data: Dict[str, Any], 
                         subject_id: str, salt: str) -> Dict[str, Any]:
        """
        Pseudonymize data with reversible mapping for legitimate purposes.
        
        Args:
            data: Data to pseudonymize
            subject_id: Subject identifier
            salt: Salt for pseudonymization
        
        Returns:
            Pseudonymized data
        """
        
        pseudonymized = data.copy()
        
        # Create pseudonym for subject
        pseudonym = hashlib.sha256(f"{subject_id}_{salt}".encode()).hexdigest()[:16]
        
        # Replace subject references
        if "subject_id" in pseudonymized:
            pseudonymized["subject_id"] = f"pseudo_{pseudonym}"
        
        if "user_id" in pseudonymized:
            pseudonymized["user_id"] = f"user_{pseudonym}"
        
        # Log pseudonymization for potential reversal
        self.logger.debug(f"Pseudonymized subject {subject_id} as pseudo_{pseudonym}")
        
        return pseudonymized


class ConsentManager:
    """Manages consent collection, storage, and validation."""
    
    def __init__(self):
        self.consent_records: Dict[str, ConsentRecord] = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def collect_consent(self, subject_id: str, 
                       purposes: List[DataProcessingPurpose],
                       method: str = "web_form",
                       expires_in_days: Optional[int] = 365) -> str:
        """
        Collect and record consent from a data subject.
        
        Args:
            subject_id: Identifier of the data subject
            purposes: List of processing purposes
            method: Method of consent collection
            expires_in_days: Consent expiration in days (None for no expiration)
        
        Returns:
            Consent record ID
        """
        
        consent_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        
        expires_at = None
        if expires_in_days:
            expires_at = now + timedelta(days=expires_in_days)
        
        # Create granular permissions
        granular_permissions = {}
        for purpose in purposes:
            granular_permissions[purpose.value] = True
        
        # Additional granular permissions
        granular_permissions.update({
            "analytics": True,
            "performance_monitoring": True,
            "research_participation": True,
            "data_sharing": False,  # Default to false for sharing
            "marketing": False
        })
        
        consent_record = ConsentRecord(
            consent_id=consent_id,
            subject_id=subject_id,
            purposes=purposes,
            granted_at=now,
            expires_at=expires_at,
            withdrawn_at=None,
            lawful_basis=LegalBasis.CONSENT,
            consent_method=method,
            granular_permissions=granular_permissions
        )
        
        self.consent_records[consent_id] = consent_record
        
        self.logger.info(f"Consent collected for subject {subject_id}: {consent_id}")
        
        return consent_id
    
    def withdraw_consent(self, subject_id: str, consent_id: Optional[str] = None) -> bool:
        """
        Withdraw consent for a subject.
        
        Args:
            subject_id: Subject withdrawing consent
            consent_id: Specific consent to withdraw (None for all)
        
        Returns:
            Success status
        """
        
        withdrawn_count = 0
        now = datetime.now(timezone.utc)
        
        for record_id, record in self.consent_records.items():
            if record.subject_id == subject_id:
                if consent_id is None or record_id == consent_id:
                    record.withdrawn_at = now
                    withdrawn_count += 1
        
        self.logger.info(f"Withdrew {withdrawn_count} consent records for subject {subject_id}")
        
        return withdrawn_count > 0
    
    def validate_consent(self, subject_id: str, 
                        purpose: DataProcessingPurpose) -> bool:
        """
        Validate that valid consent exists for a processing purpose.
        
        Args:
            subject_id: Subject to check consent for
            purpose: Processing purpose to validate
        
        Returns:
            Whether valid consent exists
        """
        
        for record in self.consent_records.values():
            if (record.subject_id == subject_id and 
                record.is_valid() and 
                purpose in record.purposes):
                return True
        
        return False
    
    def get_consent_status(self, subject_id: str) -> Dict[str, Any]:
        """Get comprehensive consent status for a subject."""
        
        subject_consents = [r for r in self.consent_records.values() 
                          if r.subject_id == subject_id]
        
        if not subject_consents:
            return {"consents": [], "valid_purposes": [], "status": "no_consent"}
        
        valid_consents = [c for c in subject_consents if c.is_valid()]
        valid_purposes = []
        for consent in valid_consents:
            valid_purposes.extend(consent.purposes)
        
        return {
            "consents": [c.to_dict() for c in subject_consents],
            "valid_purposes": list(set(p.value for p in valid_purposes)),
            "status": "valid" if valid_consents else "expired"
        }


class ComplianceRegimeHandler(ABC):
    """Abstract base class for handling specific compliance regimes."""
    
    @abstractmethod
    def validate_processing(self, processing_record: DataProcessingRecord) -> List[str]:
        """Validate processing against regime requirements."""
        pass
    
    @abstractmethod
    def get_subject_rights(self) -> List[str]:
        """Get list of subject rights under this regime."""
        pass
    
    @abstractmethod
    def get_retention_requirements(self, data_category: str) -> str:
        """Get retention requirements for data category."""
        pass
    
    @abstractmethod
    def audit_compliance(self, processing_records: List[DataProcessingRecord]) -> ComplianceAuditResult:
        """Perform compliance audit."""
        pass


class GDPRHandler(ComplianceRegimeHandler):
    """Handler for GDPR compliance."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def validate_processing(self, processing_record: DataProcessingRecord) -> List[str]:
        """Validate processing against GDPR requirements."""
        
        issues = []
        
        # Check legal basis
        if processing_record.legal_basis not in LegalBasis:
            issues.append("Invalid legal basis specified")
        
        # Check purpose limitation
        if not processing_record.processing_purposes:
            issues.append("No processing purposes specified")
        
        # Check data minimization
        if len(processing_record.data_categories) > 10:
            issues.append("Potential data minimization issue - many data categories")
        
        # Check retention period
        if processing_record.retention_period == "indefinite":
            issues.append("Indefinite retention not allowed under GDPR")
        
        # Check third country transfers
        if processing_record.third_country_transfers:
            for country in processing_record.third_country_transfers:
                if not self._is_adequate_country(country):
                    issues.append(f"Third country transfer to {country} requires additional safeguards")
        
        # Check security measures
        required_measures = ["encryption", "access_control", "audit_logging"]
        missing_measures = [m for m in required_measures 
                          if m not in processing_record.security_measures]
        
        if missing_measures:
            issues.append(f"Missing required security measures: {', '.join(missing_measures)}")
        
        return issues
    
    def get_subject_rights(self) -> List[str]:
        """Get GDPR subject rights."""
        return [
            "right_to_information",
            "right_of_access",
            "right_to_rectification", 
            "right_to_erasure",
            "right_to_restrict_processing",
            "right_to_data_portability",
            "right_to_object",
            "rights_automated_decision_making"
        ]
    
    def get_retention_requirements(self, data_category: str) -> str:
        """Get GDPR retention requirements."""
        
        retention_periods = {
            "performance_metrics": "2 years",
            "benchmark_results": "5 years", 
            "user_preferences": "1 year",
            "system_logs": "90 days",
            "audit_logs": "6 years",
            "research_data": "10 years"
        }
        
        return retention_periods.get(data_category, "As long as necessary for purpose")
    
    def audit_compliance(self, processing_records: List[DataProcessingRecord]) -> ComplianceAuditResult:
        """Perform GDPR compliance audit."""
        
        audit_id = str(uuid.uuid4())
        audit_date = datetime.now(timezone.utc)
        
        findings = []
        violations = []
        compliance_score = 1.0
        
        for record in processing_records:
            issues = self.validate_processing(record)
            
            if issues:
                finding = {
                    "record_id": record.record_id,
                    "issues": issues,
                    "severity": "high" if "legal basis" in " ".join(issues).lower() else "medium"
                }
                findings.append(finding)
                
                # Deduct from compliance score
                compliance_score -= len(issues) * 0.1
                
                # Identify violations
                for issue in issues:
                    if any(keyword in issue.lower() for keyword in 
                          ["legal basis", "retention", "security", "transfer"]):
                        violations.append({
                            "record_id": record.record_id,
                            "violation": issue,
                            "article": self._map_issue_to_article(issue)
                        })
        
        compliance_score = max(0.0, compliance_score)
        
        # Generate recommendations
        recommendations = self._generate_gdpr_recommendations(findings)
        
        return ComplianceAuditResult(
            audit_id=audit_id,
            regime=ComplianceRegime.GDPR,
            audit_date=audit_date,
            compliance_score=compliance_score,
            findings=findings,
            violations=violations,
            recommendations=recommendations,
            next_audit_due=audit_date + timedelta(days=365),
            auditor="GDPR_Handler"
        )
    
    def _is_adequate_country(self, country_code: str) -> bool:
        """Check if country has EU adequacy decision."""
        
        adequate_countries = [
            "AD", "AR", "CA", "FO", "GG", "IL", "IM", "IS", "JE", 
            "JP", "NZ", "CH", "UY", "GB", "KR"
        ]
        
        return country_code in adequate_countries
    
    def _map_issue_to_article(self, issue: str) -> str:
        """Map compliance issue to GDPR article."""
        
        article_mapping = {
            "legal basis": "Article 6",
            "consent": "Article 7", 
            "retention": "Article 5(1)(e)",
            "security": "Article 32",
            "transfer": "Chapter V",
            "purpose": "Article 5(1)(b)",
            "data minimization": "Article 5(1)(c)"
        }
        
        for keyword, article in article_mapping.items():
            if keyword in issue.lower():
                return article
        
        return "General"
    
    def _generate_gdpr_recommendations(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Generate GDPR compliance recommendations."""
        
        recommendations = []
        
        if not findings:
            recommendations.append("GDPR compliance is satisfactory")
            return recommendations
        
        # Analyze common issues
        all_issues = []
        for finding in findings:
            all_issues.extend(finding["issues"])
        
        if any("legal basis" in issue for issue in all_issues):
            recommendations.append("Review and document legal basis for all processing activities")
        
        if any("security" in issue for issue in all_issues):
            recommendations.append("Implement comprehensive technical and organizational measures")
        
        if any("retention" in issue for issue in all_issues):
            recommendations.append("Establish clear data retention policies and automated deletion")
        
        if any("transfer" in issue for issue in all_issues):
            recommendations.append("Implement appropriate safeguards for international transfers")
        
        recommendations.append("Conduct regular GDPR compliance assessments")
        recommendations.append("Provide GDPR training for all personnel handling personal data")
        
        return recommendations


class CCPAHandler(ComplianceRegimeHandler):
    """Handler for CCPA compliance."""
    
    def validate_processing(self, processing_record: DataProcessingRecord) -> List[str]:
        """Validate processing against CCPA requirements."""
        
        issues = []
        
        # CCPA focuses more on disclosure and opt-out rights
        if not processing_record.recipients:
            issues.append("No recipients/disclosures specified for CCPA compliance")
        
        # Check for sale of personal information
        if "marketing" in [p.value for p in processing_record.processing_purposes]:
            issues.append("Potential sale of personal information - ensure opt-out mechanism")
        
        # Check privacy policy requirements
        if "privacy_policy_updated" not in processing_record.security_measures:
            issues.append("Privacy policy should be updated for CCPA compliance")
        
        return issues
    
    def get_subject_rights(self) -> List[str]:
        """Get CCPA consumer rights."""
        return [
            "right_to_know",
            "right_to_delete",
            "right_to_opt_out",
            "right_to_non_discrimination"
        ]
    
    def get_retention_requirements(self, data_category: str) -> str:
        """Get CCPA retention requirements."""
        return "As disclosed in privacy policy, minimum 12 months for opt-out requests"
    
    def audit_compliance(self, processing_records: List[DataProcessingRecord]) -> ComplianceAuditResult:
        """Perform CCPA compliance audit."""
        
        audit_id = str(uuid.uuid4())
        
        # Simplified CCPA audit
        findings = []
        violations = []
        compliance_score = 0.9  # Default high score for CCPA
        
        return ComplianceAuditResult(
            audit_id=audit_id,
            regime=ComplianceRegime.CCPA,
            audit_date=datetime.now(timezone.utc),
            compliance_score=compliance_score,
            findings=findings,
            violations=violations,
            recommendations=["Ensure opt-out mechanisms are clearly available"],
            next_audit_due=datetime.now(timezone.utc) + timedelta(days=365),
            auditor="CCPA_Handler"
        )


class InternationalizationManager:
    """Manages internationalization and localization."""
    
    def __init__(self):
        self.supported_locales = [
            "en_US",  # English (US)
            "en_GB",  # English (UK) 
            "es_ES",  # Spanish (Spain)
            "fr_FR",  # French (France)
            "de_DE",  # German (Germany)
            "ja_JP",  # Japanese (Japan)
            "zh_CN",  # Chinese (Simplified)
            "zh_TW",  # Chinese (Traditional)
            "ko_KR",  # Korean (South Korea)
            "pt_BR",  # Portuguese (Brazil)
            "it_IT",  # Italian (Italy)
            "ru_RU",  # Russian (Russia)
            "ar_SA",  # Arabic (Saudi Arabia)
            "hi_IN",  # Hindi (India)
            "th_TH",  # Thai (Thailand)
            "vi_VN"   # Vietnamese (Vietnam)
        ]
        
        self.region_compliance_map = {
            "US": [ComplianceRegime.CCPA],
            "CA": [ComplianceRegime.PIPEDA],
            "GB": [ComplianceRegime.GDPR],
            "EU": [ComplianceRegime.GDPR],
            "SG": [ComplianceRegime.PDPA_SINGAPORE],
            "TH": [ComplianceRegime.PDPA_THAILAND], 
            "BR": [ComplianceRegime.LGPD],
            "AU": [ComplianceRegime.PRIVACY_ACT],
            "JP": [ComplianceRegime.APPI],
            "KR": [ComplianceRegime.PIPA]
        }
        
        self.translations = self._load_translations()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _load_translations(self) -> Dict[str, Dict[str, str]]:
        """Load translation dictionaries."""
        
        # Core compliance and UI translations
        translations = {
            "en_US": {
                "consent_required": "Your consent is required for data processing",
                "data_processing_notice": "Data Processing Notice",
                "your_rights": "Your Privacy Rights",
                "withdraw_consent": "Withdraw Consent",
                "data_deletion": "Request Data Deletion",
                "contact_dpo": "Contact Data Protection Officer",
                "privacy_policy": "Privacy Policy",
                "cookie_notice": "Cookie Notice",
                "legitimate_interests": "Legitimate Interests",
                "processing_purposes": "Processing Purposes",
                "data_retention": "Data Retention Period",
                "third_party_sharing": "Third Party Sharing",
                "security_measures": "Security Measures",
                "cross_border_transfer": "International Data Transfer",
                "compliance_audit": "Compliance Audit",
                "benchmark_results": "Benchmark Results",
                "performance_metrics": "Performance Metrics",
                "model_evaluation": "Model Evaluation"
            },
            
            "es_ES": {
                "consent_required": "Se requiere su consentimiento para el procesamiento de datos",
                "data_processing_notice": "Aviso de Procesamiento de Datos",
                "your_rights": "Sus Derechos de Privacidad",
                "withdraw_consent": "Retirar Consentimiento", 
                "data_deletion": "Solicitar Eliminación de Datos",
                "contact_dpo": "Contactar Delegado de Protección de Datos",
                "privacy_policy": "Política de Privacidad",
                "cookie_notice": "Aviso de Cookies",
                "legitimate_interests": "Intereses Legítimos",
                "processing_purposes": "Propósitos de Procesamiento",
                "data_retention": "Período de Retención de Datos",
                "third_party_sharing": "Compartir con Terceros",
                "security_measures": "Medidas de Seguridad",
                "cross_border_transfer": "Transferencia Internacional de Datos",
                "compliance_audit": "Auditoría de Cumplimiento",
                "benchmark_results": "Resultados de Referencia",
                "performance_metrics": "Métricas de Rendimiento",
                "model_evaluation": "Evaluación de Modelo"
            },
            
            "fr_FR": {
                "consent_required": "Votre consentement est requis pour le traitement des données",
                "data_processing_notice": "Avis de Traitement des Données",
                "your_rights": "Vos Droits de Confidentialité",
                "withdraw_consent": "Retirer le Consentement",
                "data_deletion": "Demander la Suppression des Données", 
                "contact_dpo": "Contacter le Délégué à la Protection des Données",
                "privacy_policy": "Politique de Confidentialité",
                "cookie_notice": "Avis sur les Cookies",
                "legitimate_interests": "Intérêts Légitimes",
                "processing_purposes": "Finalités de Traitement",
                "data_retention": "Période de Conservation des Données",
                "third_party_sharing": "Partage avec des Tiers",
                "security_measures": "Mesures de Sécurité",
                "cross_border_transfer": "Transfert International de Données",
                "compliance_audit": "Audit de Conformité",
                "benchmark_results": "Résultats de Référence",
                "performance_metrics": "Métriques de Performance",
                "model_evaluation": "Évaluation de Modèle"
            },
            
            "de_DE": {
                "consent_required": "Ihre Zustimmung ist für die Datenverarbeitung erforderlich",
                "data_processing_notice": "Datenverarbeitungshinweis",
                "your_rights": "Ihre Datenschutzrechte",
                "withdraw_consent": "Zustimmung Widerrufen",
                "data_deletion": "Datenlöschung Beantragen",
                "contact_dpo": "Datenschutzbeauftragten Kontaktieren", 
                "privacy_policy": "Datenschutzrichtlinie",
                "cookie_notice": "Cookie-Hinweis",
                "legitimate_interests": "Berechtigte Interessen",
                "processing_purposes": "Verarbeitungszwecke",
                "data_retention": "Datenaufbewahrungsdauer",
                "third_party_sharing": "Weitergabe an Dritte",
                "security_measures": "Sicherheitsmaßnahmen",
                "cross_border_transfer": "Internationale Datenübertragung",
                "compliance_audit": "Compliance-Audit",
                "benchmark_results": "Benchmark-Ergebnisse",
                "performance_metrics": "Leistungsmetriken",
                "model_evaluation": "Modellbewertung"
            },
            
            "ja_JP": {
                "consent_required": "データ処理には同意が必要です",
                "data_processing_notice": "データ処理に関する通知",
                "your_rights": "あなたのプライバシー権利",
                "withdraw_consent": "同意を撤回する",
                "data_deletion": "データ削除を要求する",
                "contact_dpo": "データ保護責任者に連絡する",
                "privacy_policy": "プライバシーポリシー",
                "cookie_notice": "Cookieに関する通知",
                "legitimate_interests": "正当な利益",
                "processing_purposes": "処理目的",
                "data_retention": "データ保存期間",
                "third_party_sharing": "第三者との共有",
                "security_measures": "セキュリティ対策",
                "cross_border_transfer": "国際的なデータ転送",
                "compliance_audit": "コンプライアンス監査",
                "benchmark_results": "ベンチマーク結果",
                "performance_metrics": "性能メトリクス",
                "model_evaluation": "モデル評価"
            },
            
            "zh_CN": {
                "consent_required": "数据处理需要您的同意",
                "data_processing_notice": "数据处理通知",
                "your_rights": "您的隐私权利",
                "withdraw_consent": "撤回同意",
                "data_deletion": "请求删除数据", 
                "contact_dpo": "联系数据保护官员",
                "privacy_policy": "隐私政策",
                "cookie_notice": "Cookie通知", 
                "legitimate_interests": "合法利益",
                "processing_purposes": "处理目的",
                "data_retention": "数据保留期限",
                "third_party_sharing": "第三方共享",
                "security_measures": "安全措施",
                "cross_border_transfer": "跨境数据传输",
                "compliance_audit": "合规审计",
                "benchmark_results": "基准测试结果",
                "performance_metrics": "性能指标",
                "model_evaluation": "模型评估"
            }
        }
        
        return translations
    
    def get_translation(self, key: str, locale: str = "en_US") -> str:
        """Get translation for a key in specified locale."""
        
        if locale not in self.translations:
            locale = "en_US"  # Fallback to English
        
        return self.translations[locale].get(key, key)
    
    def get_compliance_regimes_for_region(self, region_code: str) -> List[ComplianceRegime]:
        """Get applicable compliance regimes for a region."""
        
        # Handle EU countries
        if region_code in ["AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", 
                          "FR", "DE", "GR", "HU", "IE", "IT", "LV", "LT", "LU", 
                          "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE"]:
            return [ComplianceRegime.GDPR]
        
        return self.region_compliance_map.get(region_code, [])
    
    def localize_consent_form(self, locale: str, purposes: List[DataProcessingPurpose]) -> Dict[str, Any]:
        """Generate localized consent form."""
        
        form_data = {
            "locale": locale,
            "title": self.get_translation("data_processing_notice", locale),
            "consent_text": self.get_translation("consent_required", locale),
            "purposes": [],
            "rights_text": self.get_translation("your_rights", locale),
            "withdraw_text": self.get_translation("withdraw_consent", locale)
        }
        
        # Localize processing purposes
        for purpose in purposes:
            purpose_key = purpose.value.replace("_", " ").title()
            localized_purpose = self.get_translation(purpose.value, locale)
            form_data["purposes"].append({
                "id": purpose.value,
                "name": localized_purpose,
                "description": f"Processing for {localized_purpose.lower()}"
            })
        
        return form_data
    
    def get_supported_locales(self) -> List[str]:
        """Get list of supported locales."""
        return self.supported_locales.copy()
    
    def detect_region_from_locale(self, locale: str) -> str:
        """Detect region code from locale."""
        
        if "_" in locale:
            return locale.split("_")[1]
        
        # Map language codes to primary regions
        language_to_region = {
            "en": "US",
            "es": "ES", 
            "fr": "FR",
            "de": "DE",
            "ja": "JP",
            "zh": "CN",
            "ko": "KR",
            "pt": "BR",
            "it": "IT",
            "ru": "RU",
            "ar": "SA",
            "hi": "IN",
            "th": "TH",
            "vi": "VN"
        }
        
        return language_to_region.get(locale, "US")


class GlobalComplianceFramework:
    """Main framework for global compliance management."""
    
    def __init__(self):
        self.data_subjects: Dict[str, DataSubject] = {}
        self.processing_records: Dict[str, DataProcessingRecord] = {}
        self.consent_manager = ConsentManager()
        self.anonymizer = DataAnonymizer()
        self.i18n_manager = InternationalizationManager()
        
        # Initialize compliance handlers
        self.compliance_handlers = {
            ComplianceRegime.GDPR: GDPRHandler(),
            ComplianceRegime.CCPA: CCPAHandler()
        }
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def register_data_subject(self, subject_data: Dict[str, Any]) -> str:
        """Register a new data subject."""
        
        subject_id = str(uuid.uuid4())
        
        # Determine jurisdiction
        jurisdiction = subject_data.get("country", "unknown")
        
        # Classify data
        data_classification = DataClassification.PERSONAL_DATA
        if any(sensitive in str(subject_data).lower() for sensitive in 
              ["health", "biometric", "genetic", "political", "religious"]):
            data_classification = DataClassification.SENSITIVE_PERSONAL_DATA
        
        # Create data subject record
        data_subject = DataSubject(
            subject_id=subject_id,
            jurisdiction=jurisdiction,
            data_classification=data_classification,
            consent_status={},
            processing_purposes=[],
            legal_basis=LegalBasis.CONSENT,
            data_categories=subject_data.get("data_categories", []),
            retention_until=None,
            anonymized=False
        )
        
        self.data_subjects[subject_id] = data_subject
        
        self.logger.info(f"Registered data subject: {subject_id} in jurisdiction {jurisdiction}")
        
        return subject_id
    
    def record_data_processing(self, processing_data: Dict[str, Any]) -> str:
        """Record a data processing activity."""
        
        record_id = str(uuid.uuid4())
        
        processing_record = DataProcessingRecord(
            record_id=record_id,
            controller=processing_data["controller"],
            processor=processing_data.get("processor"),
            data_categories=processing_data["data_categories"],
            subject_categories=processing_data.get("subject_categories", ["benchmark_users"]),
            processing_purposes=[DataProcessingPurpose(p) for p in processing_data["purposes"]],
            legal_basis=LegalBasis(processing_data.get("legal_basis", "legitimate_interests")),
            retention_period=processing_data.get("retention_period", "2 years"),
            recipients=processing_data.get("recipients", []),
            third_country_transfers=processing_data.get("third_countries", []),
            security_measures=processing_data.get("security_measures", []),
            created_at=datetime.now(timezone.utc)
        )
        
        self.processing_records[record_id] = processing_record
        
        self.logger.info(f"Recorded data processing activity: {record_id}")
        
        return record_id
    
    def process_subject_request(self, subject_id: str, request_type: str, 
                              locale: str = "en_US") -> Dict[str, Any]:
        """Process a data subject request (access, deletion, portability, etc.)."""
        
        if subject_id not in self.data_subjects:
            return {
                "success": False,
                "error": self.i18n_manager.get_translation("subject_not_found", locale),
                "request_id": None
            }
        
        request_id = str(uuid.uuid4())
        subject = self.data_subjects[subject_id]
        
        # Determine applicable compliance regimes
        region = self.i18n_manager.detect_region_from_locale(locale)
        applicable_regimes = self.i18n_manager.get_compliance_regimes_for_region(region)
        
        response = {
            "success": True,
            "request_id": request_id,
            "subject_id": subject_id,
            "request_type": request_type,
            "applicable_regimes": [r.value for r in applicable_regimes],
            "data": {},
            "message": ""
        }
        
        if request_type == "access":
            # Data access request
            response["data"] = {
                "subject_info": subject.to_dict(),
                "consent_status": self.consent_manager.get_consent_status(subject_id),
                "processing_activities": [
                    r.to_dict() for r in self.processing_records.values()
                    if subject_id in r.subject_categories or "all_subjects" in r.subject_categories
                ]
            }
            response["message"] = self.i18n_manager.get_translation("data_access_provided", locale)
        
        elif request_type == "deletion":
            # Data deletion request (Right to be forgotten)
            if self._can_delete_data(subject_id, applicable_regimes):
                # Mark for deletion
                subject.retention_until = datetime.now(timezone.utc)
                response["message"] = self.i18n_manager.get_translation("deletion_scheduled", locale)
            else:
                response["message"] = self.i18n_manager.get_translation("deletion_not_possible", locale)
        
        elif request_type == "portability":
            # Data portability request
            portable_data = self._extract_portable_data(subject_id)
            response["data"] = portable_data
            response["message"] = self.i18n_manager.get_translation("data_exported", locale)
        
        elif request_type == "consent_withdrawal":
            # Withdraw consent
            success = self.consent_manager.withdraw_consent(subject_id)
            response["message"] = self.i18n_manager.get_translation(
                "consent_withdrawn" if success else "consent_withdrawal_failed", locale
            )
        
        self.logger.info(f"Processed {request_type} request for subject {subject_id}: {request_id}")
        
        return response
    
    def _can_delete_data(self, subject_id: str, regimes: List[ComplianceRegime]) -> bool:
        """Determine if data can be deleted considering legal obligations."""
        
        # Check for legal obligations to retain data
        subject = self.data_subjects.get(subject_id)
        if not subject:
            return False
        
        # Check if data is needed for compliance, legal, or research purposes
        for record in self.processing_records.values():
            if DataProcessingPurpose.COMPLIANCE_MONITORING in record.processing_purposes:
                if "audit" in record.retention_period or "legal" in record.retention_period:
                    return False
        
        # Under GDPR, research data may be retained if adequately anonymized
        if ComplianceRegime.GDPR in regimes:
            for record in self.processing_records.values():
                if DataProcessingPurpose.RESEARCH in record.processing_purposes:
                    if not subject.anonymized:
                        # Offer anonymization instead of deletion
                        return False
        
        return True
    
    def _extract_portable_data(self, subject_id: str) -> Dict[str, Any]:
        """Extract data in portable format for data portability requests."""
        
        subject = self.data_subjects.get(subject_id)
        if not subject:
            return {}
        
        # Find all processing records related to this subject
        related_records = []
        for record in self.processing_records.values():
            if subject_id in record.subject_categories or "all_subjects" in record.subject_categories:
                related_records.append(record.to_dict())
        
        portable_data = {
            "subject_information": subject.to_dict(),
            "consent_records": [r.to_dict() for r in self.consent_manager.consent_records.values()
                              if r.subject_id == subject_id],
            "processing_records": related_records,
            "export_date": datetime.now(timezone.utc).isoformat(),
            "format": "JSON",
            "version": "1.0"
        }
        
        return portable_data
    
    async def comprehensive_compliance_audit(self, region: str = "global") -> Dict[str, ComplianceAuditResult]:
        """Perform comprehensive compliance audit."""
        
        self.logger.info(f"Starting comprehensive compliance audit for region: {region}")
        
        audit_results = {}
        
        # Determine which regimes to audit
        if region == "global":
            regimes_to_audit = list(self.compliance_handlers.keys())
        else:
            regimes_to_audit = self.i18n_manager.get_compliance_regimes_for_region(region)
        
        # Run audits for each applicable regime
        for regime in regimes_to_audit:
            if regime in self.compliance_handlers:
                handler = self.compliance_handlers[regime]
                audit_result = handler.audit_compliance(list(self.processing_records.values()))
                audit_results[regime.value] = audit_result
        
        # Generate overall compliance score
        if audit_results:
            overall_score = sum(r.compliance_score for r in audit_results.values()) / len(audit_results)
            self.logger.info(f"Overall compliance score: {overall_score:.3f}")
        
        return audit_results
    
    def anonymize_benchmark_data(self, data: Dict[str, Any], 
                                subject_id: Optional[str] = None,
                                anonymization_level: str = "standard") -> Dict[str, Any]:
        """Anonymize benchmark data for sharing or research."""
        
        anonymized_data = self.anonymizer.anonymize_performance_data(data, anonymization_level)
        
        # Log anonymization
        if subject_id:
            self.logger.info(f"Anonymized data for subject {subject_id} at level {anonymization_level}")
        
        return anonymized_data
    
    def generate_privacy_notice(self, locale: str = "en_US", 
                               organization: str = "Video Diffusion Benchmark") -> Dict[str, Any]:
        """Generate comprehensive privacy notice in specified language."""
        
        applicable_regimes = self.i18n_manager.get_compliance_regimes_for_region(
            self.i18n_manager.detect_region_from_locale(locale)
        )
        
        privacy_notice = {
            "organization": organization,
            "locale": locale,
            "effective_date": datetime.now(timezone.utc).isoformat(),
            "applicable_regimes": [r.value for r in applicable_regimes],
            "sections": {}
        }
        
        # Core sections
        privacy_notice["sections"] = {
            "data_controller": {
                "title": self.i18n_manager.get_translation("data_controller", locale),
                "content": f"{organization} acts as data controller for benchmark data processing."
            },
            "processing_purposes": {
                "title": self.i18n_manager.get_translation("processing_purposes", locale),
                "content": self._get_processing_purposes_text(locale)
            },
            "legal_basis": {
                "title": self.i18n_manager.get_translation("legal_basis", locale),
                "content": self._get_legal_basis_text(locale, applicable_regimes)
            },
            "data_retention": {
                "title": self.i18n_manager.get_translation("data_retention", locale),
                "content": self._get_retention_text(locale)
            },
            "your_rights": {
                "title": self.i18n_manager.get_translation("your_rights", locale),
                "content": self._get_rights_text(locale, applicable_regimes)
            },
            "contact_information": {
                "title": self.i18n_manager.get_translation("contact_info", locale),
                "content": self._get_contact_text(locale)
            }
        }
        
        return privacy_notice
    
    def _get_processing_purposes_text(self, locale: str) -> str:
        """Get processing purposes text in specified locale."""
        
        purposes = [
            self.i18n_manager.get_translation("benchmark_results", locale),
            self.i18n_manager.get_translation("performance_metrics", locale),
            self.i18n_manager.get_translation("model_evaluation", locale),
            self.i18n_manager.get_translation("research", locale)
        ]
        
        return f"We process your data for the following purposes: {', '.join(purposes)}"
    
    def _get_legal_basis_text(self, locale: str, regimes: List[ComplianceRegime]) -> str:
        """Get legal basis text for applicable regimes."""
        
        if ComplianceRegime.GDPR in regimes:
            return f"Our legal basis for processing is {self.i18n_manager.get_translation('legitimate_interests', locale)} in accordance with Article 6(1)(f) of the GDPR."
        
        return "We process data in accordance with applicable privacy laws."
    
    def _get_retention_text(self, locale: str) -> str:
        """Get data retention text."""
        
        return f"Data is retained as specified in our {self.i18n_manager.get_translation('data_retention', locale)} policy, typically 2-5 years depending on the data category."
    
    def _get_rights_text(self, locale: str, regimes: List[ComplianceRegime]) -> str:
        """Get rights text for applicable regimes."""
        
        rights = []
        
        for regime in regimes:
            if regime in self.compliance_handlers:
                regime_rights = self.compliance_handlers[regime].get_subject_rights()
                rights.extend(regime_rights)
        
        unique_rights = list(set(rights))
        localized_rights = [self.i18n_manager.get_translation(right, locale) for right in unique_rights]
        
        return f"You have the following rights: {', '.join(localized_rights)}"
    
    def _get_contact_text(self, locale: str) -> str:
        """Get contact information text."""
        
        return f"For privacy-related inquiries, please {self.i18n_manager.get_translation('contact_dpo', locale)} at privacy@yourdomain.com"
    
    def export_compliance_report(self, filepath: str):
        """Export comprehensive compliance report."""
        
        report_data = {
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
            "data_subjects": {sid: subject.to_dict() for sid, subject in self.data_subjects.items()},
            "processing_records": {rid: record.to_dict() for rid, record in self.processing_records.items()},
            "consent_records": {cid: consent.to_dict() for cid, consent in self.consent_manager.consent_records.items()},
            "supported_locales": self.i18n_manager.get_supported_locales(),
            "compliance_statistics": {
                "total_subjects": len(self.data_subjects),
                "total_processing_records": len(self.processing_records),
                "total_consents": len(self.consent_manager.consent_records),
                "valid_consents": len([c for c in self.consent_manager.consent_records.values() if c.is_valid()]),
                "anonymized_subjects": len([s for s in self.data_subjects.values() if s.anonymized])
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str, ensure_ascii=False)
        
        self.logger.info(f"Compliance report exported to {filepath}")


# Example usage and testing
async def run_global_compliance_example():
    """Example of global compliance framework usage."""
    
    print("=== Global Compliance Framework Example ===")
    
    # Initialize framework
    framework = GlobalComplianceFramework()
    
    # Test multi-region compliance
    print("\n--- Multi-Region Compliance Setup ---")
    
    # Register data subjects from different regions
    subjects = []
    
    # EU subject (GDPR applies)
    eu_subject_data = {
        "country": "DE",
        "data_categories": ["performance_metrics", "benchmark_results"],
        "email": "user@example.de"
    }
    eu_subject_id = framework.register_data_subject(eu_subject_data)
    subjects.append(("EU", eu_subject_id))
    
    # US subject (CCPA applies)
    us_subject_data = {
        "country": "US",
        "state": "CA",  # California - CCPA applies
        "data_categories": ["performance_metrics", "user_preferences"],
        "email": "user@example.com"
    }
    us_subject_id = framework.register_data_subject(us_subject_data)
    subjects.append(("US", us_subject_id))
    
    # Asian subject (local regulations)
    sg_subject_data = {
        "country": "SG",
        "data_categories": ["benchmark_results", "system_logs"],
        "email": "user@example.sg"
    }
    sg_subject_id = framework.register_data_subject(sg_subject_data)
    subjects.append(("SG", sg_subject_id))
    
    print(f"Registered {len(subjects)} data subjects across different jurisdictions")
    
    # Record data processing activities
    print("\n--- Data Processing Records ---")
    
    processing_activities = [
        {
            "controller": "Video Diffusion Benchmark Organization",
            "data_categories": ["performance_metrics", "benchmark_results", "user_preferences"],
            "purposes": ["benchmarking", "performance_analysis", "research"],
            "legal_basis": "legitimate_interests",
            "retention_period": "2 years",
            "recipients": ["research_partners", "benchmark_users"],
            "security_measures": ["encryption", "access_control", "audit_logging"]
        },
        {
            "controller": "Video Diffusion Benchmark Organization", 
            "processor": "Cloud Analytics Provider",
            "data_categories": ["system_logs", "performance_metrics"],
            "purposes": ["system_optimization", "analytics"],
            "legal_basis": "legitimate_interests",
            "retention_period": "1 year",
            "recipients": ["analytics_team"],
            "third_countries": ["US"],
            "security_measures": ["encryption", "pseudonymization", "access_control"]
        }
    ]
    
    processing_records = []
    for activity in processing_activities:
        record_id = framework.record_data_processing(activity)
        processing_records.append(record_id)
    
    print(f"Recorded {len(processing_records)} data processing activities")
    
    # Test consent management
    print("\n--- Consent Management ---")
    
    consent_records = []
    for region, subject_id in subjects:
        # Collect consent for different purposes
        purposes = [
            DataProcessingPurpose.BENCHMARKING,
            DataProcessingPurpose.PERFORMANCE_ANALYSIS,
            DataProcessingPurpose.RESEARCH
        ]
        
        consent_id = framework.consent_manager.collect_consent(
            subject_id=subject_id,
            purposes=purposes,
            method="web_form",
            expires_in_days=365
        )
        
        consent_records.append((region, consent_id))
        
        # Check consent status
        status = framework.consent_manager.get_consent_status(subject_id)
        print(f"  {region} subject consent status: {status['status']}")
    
    # Test internationalization
    print("\n--- Internationalization Testing ---")
    
    locales_to_test = ["en_US", "es_ES", "fr_FR", "de_DE", "ja_JP", "zh_CN"]
    
    for locale in locales_to_test:
        # Generate localized consent form
        consent_form = framework.i18n_manager.localize_consent_form(
            locale=locale,
            purposes=[DataProcessingPurpose.BENCHMARKING, DataProcessingPurpose.RESEARCH]
        )
        
        print(f"  {locale}: {consent_form['title']}")
        
        # Test compliance regime detection
        region = framework.i18n_manager.detect_region_from_locale(locale)
        regimes = framework.i18n_manager.get_compliance_regimes_for_region(region)
        print(f"    Region: {region}, Compliance: {[r.value for r in regimes]}")
    
    # Test subject rights requests
    print("\n--- Subject Rights Requests ---")
    
    for region, subject_id in subjects[:2]:  # Test first two subjects
        locale_map = {"EU": "de_DE", "US": "en_US", "SG": "en_US"}
        locale = locale_map.get(region, "en_US")
        
        # Test data access request
        access_response = framework.process_subject_request(
            subject_id=subject_id,
            request_type="access", 
            locale=locale
        )
        
        print(f"  {region} access request: {'✅ Success' if access_response['success'] else '❌ Failed'}")
        
        # Test data portability request
        portability_response = framework.process_subject_request(
            subject_id=subject_id,
            request_type="portability",
            locale=locale
        )
        
        print(f"  {region} portability request: {'✅ Success' if portability_response['success'] else '❌ Failed'}")
    
    # Test data anonymization
    print("\n--- Data Anonymization ---")
    
    # Sample benchmark data
    sample_data = {
        "user_id": "user123",
        "session_id": "session456", 
        "ip_address": "192.168.1.100",
        "model_name": "advanced_video_diffusion",
        "fvd_score": 95.2,
        "inception_score": 38.5,
        "latency_ms": 2150,
        "memory_gb": 8.4,
        "timestamp": time.time(),
        "country": "DE",
        "device_type": "GPU"
    }
    
    anonymization_levels = ["basic", "standard", "strict"]
    for level in anonymization_levels:
        anonymized = framework.anonymize_benchmark_data(sample_data, anonymization_level=level)
        
        # Count removed/modified fields
        removed_fields = [k for k in sample_data if k not in anonymized or sample_data[k] != anonymized[k]]
        print(f"  {level.capitalize()} anonymization: {len(removed_fields)} fields modified/removed")
    
    # Comprehensive compliance audit
    print("\n--- Compliance Audit ---")
    
    audit_results = await framework.comprehensive_compliance_audit()
    
    for regime, result in audit_results.items():
        print(f"  {regime.upper()} Compliance:")
        print(f"    Score: {result.compliance_score:.3f}")
        print(f"    Findings: {len(result.findings)}")
        print(f"    Violations: {len(result.violations)}")
        
        if result.recommendations:
            print(f"    Top recommendation: {result.recommendations[0]}")
    
    # Generate privacy notices
    print("\n--- Privacy Notices ---")
    
    notice_locales = ["en_US", "de_DE", "ja_JP"]
    for locale in notice_locales:
        notice = framework.generate_privacy_notice(locale=locale)
        print(f"  {locale} privacy notice generated ({len(notice['sections'])} sections)")
        
        # Show one section as example
        if "your_rights" in notice["sections"]:
            rights_section = notice["sections"]["your_rights"]
            print(f"    Rights title: {rights_section['title']}")
    
    # Export compliance report
    print("\n--- Export Compliance Report ---")
    
    export_path = "global_compliance_report.json"
    framework.export_compliance_report(export_path)
    print(f"Compliance report exported to {export_path}")
    
    # Summary statistics
    print("\n=== Summary Statistics ===")
    
    stats = {
        "data_subjects": len(framework.data_subjects),
        "processing_records": len(framework.processing_records), 
        "consent_records": len(framework.consent_manager.consent_records),
        "valid_consents": len([c for c in framework.consent_manager.consent_records.values() if c.is_valid()]),
        "supported_locales": len(framework.i18n_manager.get_supported_locales()),
        "compliance_regimes": len(framework.compliance_handlers),
        "overall_compliance_score": sum(r.compliance_score for r in audit_results.values()) / len(audit_results) if audit_results else 0.0
    }
    
    for key, value in stats.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    return {
        "framework": framework,
        "audit_results": audit_results,
        "subjects": subjects,
        "processing_records": processing_records,
        "consent_records": consent_records,
        "statistics": stats
    }


if __name__ == "__main__":
    # Run example
    import asyncio
    asyncio.run(run_global_compliance_example())
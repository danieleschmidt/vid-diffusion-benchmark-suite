"""Universal compliance framework for global video diffusion deployments.

Advanced compliance system that ensures adherence to international data
protection regulations, industry standards, and regional requirements.
"""

import asyncio
import time
import json
import logging
import hashlib
import uuid
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)


class ComplianceFramework(Enum):
    """Supported compliance frameworks."""
    GDPR = "gdpr"  # EU General Data Protection Regulation
    CCPA = "ccpa"  # California Consumer Privacy Act
    PDPA = "pdpa"  # Personal Data Protection Act (Singapore)
    LGPD = "lgpd"  # Lei Geral de Proteção de Dados (Brazil)
    PIPEDA = "pipeda"  # Personal Information Protection and Electronic Documents Act (Canada)
    KVKK = "kvkk"  # Kişisel Verilerin Korunması Kanunu (Turkey)
    SOX = "sox"    # Sarbanes-Oxley Act
    HIPAA = "hipaa"  # Health Insurance Portability and Accountability Act
    ISO27001 = "iso27001"  # Information Security Management
    SOC2 = "soc2"  # Service Organization Control 2


class DataClassification(Enum):
    """Data classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    PERSONAL = "personal"
    SENSITIVE_PERSONAL = "sensitive_personal"


class ProcessingLawfulness(Enum):
    """Legal basis for data processing under GDPR."""
    CONSENT = "consent"
    CONTRACT = "contract"
    LEGAL_OBLIGATION = "legal_obligation"
    VITAL_INTERESTS = "vital_interests"
    PUBLIC_TASK = "public_task"
    LEGITIMATE_INTERESTS = "legitimate_interests"


@dataclass
class DataSubject:
    """Individual whose personal data is being processed."""
    subject_id: str
    region: str
    applicable_frameworks: List[ComplianceFramework]
    consent_status: Dict[str, bool] = field(default_factory=dict)
    consent_timestamp: Dict[str, datetime] = field(default_factory=dict)
    data_portability_requests: List[str] = field(default_factory=list)
    deletion_requests: List[str] = field(default_factory=list)
    opt_out_preferences: Dict[str, bool] = field(default_factory=dict)


@dataclass
class DataProcessingRecord:
    """Record of data processing activity."""
    record_id: str
    processing_purpose: str
    data_categories: List[str]
    data_classification: DataClassification
    legal_basis: ProcessingLawfulness
    data_subjects: List[str]
    retention_period: timedelta
    processing_start: datetime
    processing_end: Optional[datetime] = None
    third_party_transfers: List[str] = field(default_factory=list)
    security_measures: List[str] = field(default_factory=list)
    automated_decision_making: bool = False


@dataclass
class PrivacyImpactAssessment:
    """Privacy Impact Assessment (PIA) for high-risk processing."""
    assessment_id: str
    processing_description: str
    necessity_assessment: str
    proportionality_assessment: str
    risk_level: str  # "low", "medium", "high"
    identified_risks: List[str]
    mitigation_measures: List[str]
    residual_risks: List[str]
    approval_status: str
    reviewer: str
    assessment_date: datetime


class UniversalComplianceManager:
    """Comprehensive compliance management system."""
    
    def __init__(self):
        self.data_subjects: Dict[str, DataSubject] = {}
        self.processing_records: Dict[str, DataProcessingRecord] = {}
        self.privacy_assessments: Dict[str, PrivacyImpactAssessment] = {}
        
        # Compliance configurations
        self.framework_configs = self._initialize_framework_configs()
        self.retention_policies = self._initialize_retention_policies()
        self.security_controls = self._initialize_security_controls()
        
        # Audit trail
        self.audit_log: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        
        # Automated compliance monitoring
        self.monitoring_enabled = True
        self._monitoring_task: Optional[asyncio.Task] = None
        
    def _initialize_framework_configs(self) -> Dict[ComplianceFramework, Dict[str, Any]]:
        """Initialize configuration for each compliance framework."""
        return {
            ComplianceFramework.GDPR: {
                "data_protection_officer_required": True,
                "consent_management_required": True,
                "right_to_be_forgotten": True,
                "data_portability": True,
                "breach_notification_hours": 72,
                "privacy_by_design": True,
                "territorial_scope": ["EU", "EEA"],
                "maximum_fine": "4% of global revenue or €20M",
                "key_principles": [
                    "lawfulness", "fairness", "transparency",
                    "purpose_limitation", "data_minimization",
                    "accuracy", "storage_limitation",
                    "integrity_confidentiality", "accountability"
                ]
            },
            ComplianceFramework.CCPA: {
                "consumer_rights": [
                    "right_to_know", "right_to_delete", 
                    "right_to_opt_out", "right_to_non_discrimination"
                ],
                "data_sale_restrictions": True,
                "opt_out_required": True,
                "territorial_scope": ["California"],
                "revenue_threshold": 25000000,
                "consumer_threshold": 50000,
                "verification_requirements": True
            },
            ComplianceFramework.PDPA: {
                "consent_requirements": "explicit",
                "data_breach_notification": True,
                "territorial_scope": ["Singapore"],
                "data_protection_officer": True,
                "cross_border_transfer_restrictions": True
            },
            ComplianceFramework.SOC2: {
                "trust_criteria": [
                    "security", "availability", "processing_integrity",
                    "confidentiality", "privacy"
                ],
                "audit_requirements": "annual",
                "control_objectives": True,
                "vendor_management": True
            }
        }
        
    def _initialize_retention_policies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize data retention policies."""
        return {
            "user_analytics": {
                "retention_period": timedelta(days=365 * 2),  # 2 years
                "applicable_frameworks": [ComplianceFramework.GDPR, ComplianceFramework.CCPA],
                "auto_deletion": True,
                "review_cycle": timedelta(days=180)
            },
            "benchmark_results": {
                "retention_period": timedelta(days=365 * 7),  # 7 years
                "applicable_frameworks": [ComplianceFramework.SOX],
                "auto_deletion": False,
                "review_cycle": timedelta(days=365)
            },
            "model_training_data": {
                "retention_period": timedelta(days=365 * 3),  # 3 years
                "applicable_frameworks": [ComplianceFramework.GDPR],
                "auto_deletion": True,
                "anonymization_required": True
            },
            "audit_logs": {
                "retention_period": timedelta(days=365 * 7),  # 7 years
                "applicable_frameworks": [ComplianceFramework.SOX, ComplianceFramework.SOC2],
                "auto_deletion": False,
                "immutable": True
            }
        }
        
    def _initialize_security_controls(self) -> Dict[str, Dict[str, Any]]:
        """Initialize security control framework."""
        return {
            "encryption": {
                "at_rest": {
                    "algorithm": "AES-256",
                    "key_management": "HSM",
                    "required_frameworks": [ComplianceFramework.GDPR, ComplianceFramework.HIPAA]
                },
                "in_transit": {
                    "protocol": "TLS 1.3",
                    "certificate_validation": True,
                    "required_frameworks": [ComplianceFramework.SOC2]
                }
            },
            "access_control": {
                "authentication": {
                    "multi_factor": True,
                    "password_policy": "complex",
                    "session_timeout": 3600
                },
                "authorization": {
                    "principle": "least_privilege",
                    "role_based": True,
                    "regular_review": True
                }
            },
            "monitoring": {
                "activity_logging": True,
                "anomaly_detection": True,
                "real_time_alerts": True,
                "log_integrity": True
            }
        }
        
    async def register_data_subject(
        self,
        subject_id: str,
        region: str,
        applicable_frameworks: List[ComplianceFramework]
    ) -> DataSubject:
        """Register a new data subject for compliance tracking."""
        
        data_subject = DataSubject(
            subject_id=subject_id,
            region=region,
            applicable_frameworks=applicable_frameworks
        )
        
        with self._lock:
            self.data_subjects[subject_id] = data_subject
            
        await self._log_audit_event({
            "event_type": "data_subject_registered",
            "subject_id": subject_id,
            "region": region,
            "frameworks": [f.value for f in applicable_frameworks],
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        logger.info(f"Registered data subject {subject_id} for region {region}")
        return data_subject
        
    async def record_consent(
        self,
        subject_id: str,
        purpose: str,
        consent_given: bool,
        consent_mechanism: str = "explicit"
    ) -> bool:
        """Record consent for data processing."""
        
        if subject_id not in self.data_subjects:
            raise ValueError(f"Data subject {subject_id} not registered")
            
        data_subject = self.data_subjects[subject_id]
        
        with self._lock:
            data_subject.consent_status[purpose] = consent_given
            data_subject.consent_timestamp[purpose] = datetime.now(timezone.utc)
            
        await self._log_audit_event({
            "event_type": "consent_recorded",
            "subject_id": subject_id,
            "purpose": purpose,
            "consent_given": consent_given,
            "mechanism": consent_mechanism,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Validate consent requirements for applicable frameworks
        await self._validate_consent_compliance(data_subject, purpose, consent_given)
        
        return True
        
    async def create_processing_record(
        self,
        purpose: str,
        data_categories: List[str],
        data_classification: DataClassification,
        legal_basis: ProcessingLawfulness,
        data_subjects: List[str],
        retention_period: timedelta,
        security_measures: List[str],
        automated_decision_making: bool = False
    ) -> DataProcessingRecord:
        """Create a data processing activity record."""
        
        record_id = str(uuid.uuid4())
        
        processing_record = DataProcessingRecord(
            record_id=record_id,
            processing_purpose=purpose,
            data_categories=data_categories,
            data_classification=data_classification,
            legal_basis=legal_basis,
            data_subjects=data_subjects,
            retention_period=retention_period,
            processing_start=datetime.now(timezone.utc),
            security_measures=security_measures,
            automated_decision_making=automated_decision_making
        )
        
        with self._lock:
            self.processing_records[record_id] = processing_record
            
        await self._log_audit_event({
            "event_type": "processing_record_created",
            "record_id": record_id,
            "purpose": purpose,
            "data_classification": data_classification.value,
            "legal_basis": legal_basis.value,
            "subject_count": len(data_subjects),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Trigger privacy impact assessment if needed
        if await self._requires_privacy_impact_assessment(processing_record):
            await self._initiate_privacy_impact_assessment(processing_record)
            
        return processing_record
        
    async def handle_data_subject_request(
        self,
        subject_id: str,
        request_type: str,
        request_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle data subject rights requests."""
        
        if subject_id not in self.data_subjects:
            raise ValueError(f"Data subject {subject_id} not registered")
            
        data_subject = self.data_subjects[subject_id]
        
        if request_type == "access":
            return await self._handle_access_request(data_subject)
        elif request_type == "portability":
            return await self._handle_portability_request(data_subject)
        elif request_type == "deletion":
            return await self._handle_deletion_request(data_subject, request_details)
        elif request_type == "rectification":
            return await self._handle_rectification_request(data_subject, request_details)
        elif request_type == "opt_out":
            return await self._handle_opt_out_request(data_subject, request_details)
        else:
            raise ValueError(f"Unsupported request type: {request_type}")
            
    async def _handle_access_request(self, data_subject: DataSubject) -> Dict[str, Any]:
        """Handle subject access request (SAR)."""
        
        # Compile all data related to the subject
        subject_data = {
            "subject_id": data_subject.subject_id,
            "region": data_subject.region,
            "consent_records": data_subject.consent_status,
            "consent_timestamps": {
                k: v.isoformat() for k, v in data_subject.consent_timestamp.items()
            },
            "processing_activities": [],
            "data_transfers": [],
            "automated_decisions": []
        }
        
        # Find all processing activities involving this subject
        for record in self.processing_records.values():
            if data_subject.subject_id in record.data_subjects:
                subject_data["processing_activities"].append({
                    "purpose": record.processing_purpose,
                    "legal_basis": record.legal_basis.value,
                    "data_categories": record.data_categories,
                    "retention_period": str(record.retention_period),
                    "start_date": record.processing_start.isoformat()
                })
                
        await self._log_audit_event({
            "event_type": "subject_access_request",
            "subject_id": data_subject.subject_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return subject_data
        
    async def _handle_deletion_request(
        self,
        data_subject: DataSubject,
        request_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle right to be forgotten (erasure) request."""
        
        # Check if deletion is permissible under applicable frameworks
        deletion_permitted = await self._validate_deletion_request(data_subject, request_details)
        
        if not deletion_permitted["permitted"]:
            return {
                "status": "denied",
                "reason": deletion_permitted["reason"],
                "legal_basis": deletion_permitted["legal_basis"]
            }
            
        # Perform data deletion
        deleted_data = []
        
        # Mark processing records for deletion
        for record_id, record in self.processing_records.items():
            if data_subject.subject_id in record.data_subjects:
                # Check if retention period allows deletion
                if datetime.now(timezone.utc) > record.processing_start + record.retention_period:
                    deleted_data.append(f"processing_record_{record_id}")
                    
        # Add to deletion queue
        data_subject.deletion_requests.append(str(uuid.uuid4()))
        
        await self._log_audit_event({
            "event_type": "deletion_request_processed",
            "subject_id": data_subject.subject_id,
            "deleted_items": len(deleted_data),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        return {
            "status": "completed",
            "deleted_items": deleted_data,
            "completion_date": datetime.now(timezone.utc).isoformat()
        }
        
    async def check_compliance_status(
        self,
        framework: ComplianceFramework
    ) -> Dict[str, Any]:
        """Check overall compliance status for a framework."""
        
        compliance_checks = {
            ComplianceFramework.GDPR: self._check_gdpr_compliance,
            ComplianceFramework.CCPA: self._check_ccpa_compliance,
            ComplianceFramework.SOC2: self._check_soc2_compliance
        }
        
        if framework not in compliance_checks:
            return {"status": "not_supported", "framework": framework.value}
            
        return await compliance_checks[framework]()
        
    async def _check_gdpr_compliance(self) -> Dict[str, Any]:
        """Check GDPR compliance status."""
        
        compliance_items = []
        score = 0
        max_score = 0
        
        # Check consent management
        max_score += 1
        consent_compliant = all(
            len(ds.consent_status) > 0 for ds in self.data_subjects.values()
            if ComplianceFramework.GDPR in ds.applicable_frameworks
        )
        if consent_compliant:
            score += 1
            compliance_items.append({
                "requirement": "Consent Management",
                "status": "compliant",
                "details": "Valid consent recorded for all subjects"
            })
        else:
            compliance_items.append({
                "requirement": "Consent Management", 
                "status": "non_compliant",
                "details": "Missing consent records for some subjects"
            })
            
        # Check data processing records
        max_score += 1
        if len(self.processing_records) > 0:
            score += 1
            compliance_items.append({
                "requirement": "Article 30 Records",
                "status": "compliant", 
                "details": f"{len(self.processing_records)} processing activities documented"
            })
        else:
            compliance_items.append({
                "requirement": "Article 30 Records",
                "status": "non_compliant",
                "details": "No processing activities documented"
            })
            
        # Check privacy impact assessments
        max_score += 1
        high_risk_processing = sum(
            1 for record in self.processing_records.values()
            if record.data_classification in [DataClassification.SENSITIVE_PERSONAL]
        )
        pia_count = len(self.privacy_assessments)
        
        if high_risk_processing == 0 or pia_count >= high_risk_processing:
            score += 1
            compliance_items.append({
                "requirement": "Privacy Impact Assessments",
                "status": "compliant",
                "details": f"{pia_count} PIAs completed for {high_risk_processing} high-risk activities"
            })
        else:
            compliance_items.append({
                "requirement": "Privacy Impact Assessments",
                "status": "non_compliant", 
                "details": f"Missing PIAs for {high_risk_processing - pia_count} high-risk activities"
            })
            
        compliance_percentage = (score / max_score) * 100 if max_score > 0 else 0
        
        return {
            "framework": "GDPR",
            "compliance_percentage": compliance_percentage,
            "status": "compliant" if compliance_percentage >= 90 else "partial" if compliance_percentage >= 70 else "non_compliant",
            "items": compliance_items,
            "last_assessment": datetime.now(timezone.utc).isoformat()
        }
        
    async def generate_compliance_report(
        self,
        frameworks: List[ComplianceFramework],
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        
        report = {
            "report_id": str(uuid.uuid4()),
            "generation_date": datetime.now(timezone.utc).isoformat(),
            "frameworks": {},
            "summary": {
                "total_data_subjects": len(self.data_subjects),
                "total_processing_activities": len(self.processing_records),
                "total_privacy_assessments": len(self.privacy_assessments)
            },
            "recommendations": []
        }
        
        # Check compliance for each framework
        for framework in frameworks:
            compliance_status = await self.check_compliance_status(framework)
            report["frameworks"][framework.value] = compliance_status
            
            # Add recommendations based on compliance gaps
            if compliance_status.get("compliance_percentage", 0) < 90:
                for item in compliance_status.get("items", []):
                    if item["status"] == "non_compliant":
                        report["recommendations"].append({
                            "framework": framework.value,
                            "requirement": item["requirement"],
                            "recommendation": self._get_compliance_recommendation(framework, item["requirement"])
                        })
                        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
                
        return report
        
    def _get_compliance_recommendation(
        self,
        framework: ComplianceFramework,
        requirement: str
    ) -> str:
        """Get specific recommendation for compliance gap."""
        
        recommendations = {
            (ComplianceFramework.GDPR, "Consent Management"): 
                "Implement explicit consent collection mechanisms with clear opt-in/opt-out options",
            (ComplianceFramework.GDPR, "Article 30 Records"):
                "Document all data processing activities including purpose, legal basis, and retention periods",
            (ComplianceFramework.GDPR, "Privacy Impact Assessments"):
                "Conduct PIAs for high-risk processing activities involving sensitive personal data",
            (ComplianceFramework.CCPA, "Consumer Rights"):
                "Implement consumer request handling system for access, deletion, and opt-out rights",
            (ComplianceFramework.SOC2, "Security Controls"):
                "Implement comprehensive security controls covering all trust service criteria"
        }
        
        return recommendations.get(
            (framework, requirement),
            f"Review {framework.value} requirements for {requirement}"
        )
        
    async def _log_audit_event(self, event: Dict[str, Any]):
        """Log audit event for compliance tracking."""
        
        audit_entry = {
            "event_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **event
        }
        
        with self._lock:
            self.audit_log.append(audit_entry)
            
        # Keep audit log manageable
        if len(self.audit_log) > 10000:
            self.audit_log = self.audit_log[-5000:]
            
    async def _validate_consent_compliance(
        self,
        data_subject: DataSubject,
        purpose: str,
        consent_given: bool
    ):
        """Validate consent meets framework requirements."""
        
        for framework in data_subject.applicable_frameworks:
            if framework == ComplianceFramework.GDPR:
                # GDPR requires explicit consent for some purposes
                if not consent_given and purpose in ["marketing", "profiling"]:
                    logger.warning(
                        f"GDPR compliance risk: No consent for {purpose} from subject {data_subject.subject_id}"
                    )
                    
    async def _requires_privacy_impact_assessment(
        self,
        processing_record: DataProcessingRecord
    ) -> bool:
        """Determine if processing requires a PIA."""
        
        # High-risk indicators
        high_risk_factors = [
            processing_record.data_classification == DataClassification.SENSITIVE_PERSONAL,
            processing_record.automated_decision_making,
            len(processing_record.data_subjects) > 1000,
            len(processing_record.third_party_transfers) > 0
        ]
        
        return sum(high_risk_factors) >= 2
        
    async def _validate_deletion_request(
        self,
        data_subject: DataSubject,
        request_details: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate if deletion request can be honored."""
        
        # Check for legal obligations that prevent deletion
        legal_hold_reasons = []
        
        # Check ongoing legal proceedings
        if request_details.get("legal_proceeding", False):
            legal_hold_reasons.append("Ongoing legal proceedings")
            
        # Check regulatory retention requirements
        for record in self.processing_records.values():
            if data_subject.subject_id in record.data_subjects:
                retention_end = record.processing_start + record.retention_period
                if datetime.now(timezone.utc) < retention_end:
                    legal_hold_reasons.append(f"Retention period until {retention_end.date()}")
                    
        if legal_hold_reasons:
            return {
                "permitted": False,
                "reason": "Legal obligations prevent deletion",
                "legal_basis": legal_hold_reasons
            }
            
        return {"permitted": True}


# Global compliance manager instance
compliance_manager = UniversalComplianceManager()
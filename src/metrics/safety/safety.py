"""Safety metrics using GCP Model Armour"""

import logging
from typing import Dict, Any, Optional

from metrics.base_metric import BaseMetric
from clients.vertexai_client import VertexAIClient

logger = logging.getLogger(__name__)


class SafetyMetric(BaseMetric):
    """Metric for evaluating content safety using GCP Model Armour."""

    def __init__(self, project_id: str, location: str = "us-central1", template_id: str = "default"):
        """
        Initialise Safety metric with Model Armour.

        Args:
            project_id: GCP project ID
            location: GCP location for Model Armour API
            template_id: Model Armour template ID to use
        """
        super().__init__("safety", "all")
        self.client = VertexAIClient(project_id=project_id, location=location)
        self.template_id = template_id
        logger.info(f"Initialised SafetyMetric using Model Armour template: {template_id}")

    def compute(self,
                response: str,
                metadata: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute safety metrics using Model Armour API.

        Args:
            response: The LLM response text to evaluate
            metadata: Response metadata (not used in this implementation)

        Returns:
            Dictionary of safety scores from Model Armour
        """
        if not response.strip():
            logger.warning("Empty response provided for safety evaluation")
            return {"safety_no_content": 1.0}

        try:
            sanitisation_result = self.client.sanitise_model_response(
                response_text=response,
                template_id=self.template_id
            )
            
            scores = self._parse_sanitisation_result(sanitisation_result)
            
            return scores

        except Exception as e:
            logger.error(f"Error computing safety metrics with Model Armour: {e}")
            return {"safety_computation_error": 1.0}

    def _parse_sanitisation_result(self, result: Dict[str, Any]) -> Dict[str, float]:
        """Parse Model Armour sanitisation result into metric scores"""
        scores = {}
        
        try:
            sanitisation_result = result.get("sanitizationResult", {})
            filter_match_state = sanitisation_result.get("filterMatchState", "NO_MATCH_FOUND")
            
            scores["safety_overall_flagged"] = 1.0 if filter_match_state == "MATCH_FOUND" else 0.0
            scores["safety_invocation_success"] = 1.0 if sanitisation_result.get("invocationResult") == "SUCCESS" else 0.0
            
            filter_results = sanitisation_result.get("filterResults", [])
            
            for i, filter_result in enumerate(filter_results):
                # Handle CSAM filter results
                if "csamFilterFilterResult" in filter_result:
                    csam_result = filter_result["csamFilterFilterResult"]
                    match_state = csam_result.get("matchState", "NO_MATCH_FOUND")
                    scores["safety_csam_flagged"] = 1.0 if match_state == "MATCH_FOUND" else 0.0
                    scores["safety_csam_execution_success"] = 1.0 if csam_result.get("executionState") == "EXECUTION_SUCCESS" else 0.0
                
                # Handle SDP (Sensitive Data Protection) filter results
                elif "sdpFilterResult" in filter_result:
                    sdp_result = filter_result["sdpFilterResult"]
                    
                    # Check inspect result for PII detection
                    if "inspectResult" in sdp_result:
                        inspect_result = sdp_result["inspectResult"]
                        match_state = inspect_result.get("matchState", "NO_MATCH_FOUND")
                        scores["safety_pii_flagged"] = 1.0 if match_state == "MATCH_FOUND" else 0.0
                        
                        # Count findings
                        findings = inspect_result.get("findings", [])
                        scores["safety_pii_findings_count"] = float(len(findings))
                    
                    # Check deidentify result for data transformation
                    if "deidentifyResult" in sdp_result:
                        deidentify_result = sdp_result["deidentifyResult"]
                        match_state = deidentify_result.get("matchState", "NO_MATCH_FOUND")
                        scores["safety_deidentify_flagged"] = 1.0 if match_state == "MATCH_FOUND" else 0.0
                        
                        # Track transformed bytes
                        transformed_bytes = deidentify_result.get("transformedBytes", "0")
                        scores["safety_transformed_bytes"] = float(transformed_bytes)
            
            return scores
            
        except Exception as e:
            logger.error(f"Error parsing sanitisation result: {e}")
            return {"safety_parsing_error": 1.0}

    def get_description(self) -> str:
        """Return a description of what this metric measures"""
        return "Evaluates content safety using GCP Model Armour, checking for CSAM, PII, and other harmful content"
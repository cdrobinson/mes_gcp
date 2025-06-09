import time
from typing import Dict, Any, List, Optional, Tuple
from core.gcp_client import get_gemini_model, get_gemini_generation_config
from vertexai.generative_models import Part, GenerationConfig
from core.retry_handler import default_retry_decorator
from .base_metric import BaseMetric

class LLMJudgeEvaluator(BaseMetric):
    """
    LLM-as-a-Judge evaluator for assessing text quality across multiple dimensions.
    Uses a separate LLM instance to evaluate generated content.
    """
    
    def __init__(self, judge_llm_config_name: str = None):
        """
        Initialize LLM Judge evaluator.
        
        Args:
            judge_llm_config_name: LLM configuration to use for judging (should be different from evaluated model)
        """
        # Use a different LLM config for judging to avoid self-evaluation bias
        self.judge_llm_config_name = judge_llm_config_name or "gemini_1_5_flash_default"
        self.judge_model = get_gemini_model(self.judge_llm_config_name)
        self.judge_gen_config = get_gemini_generation_config(self.judge_llm_config_name)
        
        # Make judge more deterministic
        judge_config_dict = self.judge_gen_config.to_dict()
        judge_config_dict.update({"temperature": 0.1})  # Lower temperature for consistency
        self.judge_gen_config = GenerationConfig(**judge_config_dict)
    
    def _get_summary_evaluation_prompt(self) -> str:
        """Get the prompt template for summary evaluation."""
        return """You are an expert evaluator of text summaries. Please evaluate the following summary.

Rate each dimension on a scale of 1-5, where:
1 = Very Poor, 2 = Poor, 3 = Average, 4 = Good, 5 = Excellent

EVALUATION DIMENSIONS:
1. **Accuracy**: How factually correct is the summary? (Consider if it introduces information not present or contradicts the source, if a source is available. If no source, evaluate based on general knowledge and internal consistency.)
2. **Completeness**: How well does the summary capture the main points of the likely input?
3. **Conciseness**: How well does the summary avoid unnecessary details while maintaining clarity?
4. **Coherence**: How logically structured and easy to follow is the summary?
5. **Clarity**: How clear and understandable is the language used?

SUMMARY TO EVALUATE:
{summary}

Please provide your evaluation in the following format:
ACCURACY: [score] - [brief explanation]
COMPLETENESS: [score] - [brief explanation]  
CONCISENESS: [score] - [brief explanation]
COHERENCE: [score] - [brief explanation]
CLARITY: [score] - [brief explanation]
OVERALL: [score] - [brief overall assessment]
"""

    def _get_analysis_evaluation_prompt(self) -> str: # This can be adapted for classification or other tasks
        """Get the prompt template for a general text analysis/classification evaluation."""
        return """You are an expert evaluator of text analysis. Please evaluate the following text.

Rate each dimension on a scale of 1-5, where:
1 = Very Poor, 2 = Poor, 3 = Average, 4 = Good, 5 = Excellent

EVALUATION DIMENSIONS:
1. **Appropriateness**: How appropriate is the analysis/classification given the likely input?
2. **Confidence_Justification**: If a confidence score is provided, how well is it justified by the text? (If no confidence score, rate as N/A)
3. **Clarity_Of_Reasoning**: How clear is the reasoning behind the analysis/classification?
4. **Potential_Bias**: Does the analysis/classification show any signs of bias?
5. **Helpfulness**: How helpful is this analysis/classification?

TEXT TO EVALUATE:
{text_input}

Please provide your evaluation in the following format:
APPROPRIATENESS: [score] - [brief explanation]
CONFIDENCE_JUSTIFICATION: [score] - [brief explanation]
CLARITY_OF_REASONING: [score] - [brief explanation]
POTENTIAL_BIAS: [score] - [brief explanation]
HELPFULNESS: [score] - [brief explanation]
OVERALL: [score] - [brief overall assessment]
"""

    @default_retry_decorator()
    def _call_judge_llm(self, prompt: str) -> Tuple[str, float, Dict[str, Any]]:
        """Call the judge LLM with retry logic."""
        start_time = time.time()
        
        response = self.judge_model.generate_content(
            contents=[Part.from_text(prompt)],
            generation_config=self.judge_gen_config
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        llm_output = ""
        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.text:
                    llm_output += part.text
        
        usage_metadata = response.usage_metadata if hasattr(response, 'usage_metadata') else None
        metadata = {
            "judge_llm_config": self.judge_llm_config_name,
            "judge_processing_time": round(processing_time, 3),
            "judge_tokens_used": usage_metadata.total_token_count if usage_metadata else 0
        }
        
        return llm_output.strip(), processing_time, metadata

    def _parse_evaluation_scores(self, evaluation_text: str) -> Dict[str, Any]:
        """Parse numerical scores from the LLM judge evaluation."""
        scores = {}
        lines = evaluation_text.split('\n')
        
        dimension_mappings = {
            'ACCURACY': 'accuracy',
            'COMPLETENESS': 'completeness', 
            'CONCISENESS': 'conciseness',
            'COHERENCE': 'coherence',
            'CLARITY': 'clarity',
            'APPROPRIATENESS': 'appropriateness',
            'CONFIDENCE_JUSTIFICATION': 'confidence_justification',
            'CLARITY_OF_REASONING': 'clarity_of_reasoning',
            'POTENTIAL_BIAS': 'potential_bias',
            'HELPFULNESS': 'helpfulness',
            # Dimension for Q&A Groundedness
            'GROUNDEDNESS_SUPPORT': 'groundedness_support',
            'OVERALL': 'overall'
        }
        
        for line in lines:
            line = line.strip()
            for key, mapped_key in dimension_mappings.items():
                if line.startswith(f"{key}:"):
                    try:
                        # Extract score (first number after colon)
                        score_part = line.split(':')[1].strip()
                        score = int(score_part.split()[0])
                        if 1 <= score <= 5:  # Validate score range
                            scores[f"llm_judge_{mapped_key}"] = score
                    except:
                        continue  # Skip if parsing fails
        
        return scores

    def evaluate_summary(self, summary: str, transcript: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a summary using LLM-as-a-Judge.
        
        Args:
            summary: Generated summary to evaluate
            transcript: Optional original transcript (can be None)
            
        Returns:
            Dictionary containing LLM judge scores and metadata
        """
        if not summary or not summary.strip():
            return {
                "llm_judge_accuracy": 0,
                "llm_judge_completeness": 0,
                "llm_judge_conciseness": 0,
                "llm_judge_coherence": 0,
                "llm_judge_clarity": 0,
                "llm_judge_overall": 0,
                "llm_judge_evaluation_text": "Error: Empty summary",
                "llm_judge_metadata": {"error": "Empty summary provided"}
            }
        
        prompt_format_args = {"summary": summary}
        if transcript:
             prompt_format_args["transcript"] = transcript[:2000] # Limit length

        prompt = self._get_summary_evaluation_prompt().format(**prompt_format_args)
        
        try:
            evaluation_text, _, metadata = self._call_judge_llm(prompt)
            scores = self._parse_evaluation_scores(evaluation_text)
            
            result = {
                "llm_judge_evaluation_text": evaluation_text,
                "llm_judge_metadata": metadata,
                **scores
            }
            
            return result
            
        except Exception as e:
            # Ensure all expected keys are present in case of error
            return {
                "llm_judge_accuracy": 0,
                "llm_judge_completeness": 0,
                "llm_judge_conciseness": 0,
                "llm_judge_coherence": 0,
                "llm_judge_clarity": 0,
                "llm_judge_overall": 0,
                "llm_judge_evaluation_text": f"Error: {str(e)}",
                "llm_judge_metadata": {"error": str(e)}
            }

    def evaluate_classification(self, llm_response: str, input_text: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluate a classification output using LLM-as-a-Judge.
        
        Args:
            llm_response: The classification output from the LLM.
            input_text: Optional original input text that was classified (can be None)
            
        Returns:
            Dictionary containing LLM judge scores and metadata
        """
        if not llm_response or not llm_response.strip():
            return {
                "llm_judge_appropriateness": 0,
                "llm_judge_confidence_justification": 0,
                "llm_judge_clarity_of_reasoning": 0,
                "llm_judge_potential_bias": 0,
                "llm_judge_helpfulness": 0,
                "llm_judge_overall": 0,
                "llm_judge_evaluation_text": "Error: Empty LLM response for classification",
                "llm_judge_metadata": {"error": "Empty LLM response provided"}
            }

        prompt_format_args = {"text_input": llm_response} # Evaluate the LLM response itself
        if input_text:
            # If original input is provided, it can be added to prompt for context,
            # but the primary evaluation is on the llm_response (the classification).
            # For now, the prompt is simpler and focuses on the llm_response.
            # Modify _get_analysis_evaluation_prompt if deeper context is needed.
            pass

        prompt = self._get_analysis_evaluation_prompt().format(**prompt_format_args)
        
        try:
            evaluation_text, _, metadata = self._call_judge_llm(prompt)
            scores = self._parse_evaluation_scores(evaluation_text)
            
            result = {
                "llm_judge_evaluation_text": evaluation_text,
                "llm_judge_metadata": metadata,
                **scores
            }
            
            return result
            
        except Exception as e:
            return {
                "llm_judge_appropriateness": 0,
                "llm_judge_confidence_justification": 0,
                "llm_judge_clarity_of_reasoning": 0,
                "llm_judge_potential_bias": 0,
                "llm_judge_helpfulness": 0,
                "llm_judge_overall": 0,
                "llm_judge_evaluation_text": f"Error: {str(e)}",
                "llm_judge_metadata": {"error": str(e)}
            }

    def _get_question_generation_prompt(self, summary: str) -> str:
        """Prompt to generate questions based on the summary."""
        return f"""Given the following summary, please generate 3-5 factual questions that should be answerable *only* using the information explicitly provided in the summary.
Focus on key pieces of information, claims, or entities mentioned. Avoid questions requiring external knowledge.

SUMMARY:
{summary}

QUESTIONS (one per line):
"""

    def _get_answer_generation_prompt(self, summary: str, questions: List[str]) -> str:
        """Prompt to answer generated questions based *only* on the summary."""
        question_block = "\\n".join([f"Q: {q}" for q in questions])
        return f"""Please answer the following questions based *solely* on the provided summary.
If the answer is not found in the summary, explicitly state "Information not found in summary". Do not infer or use external knowledge.

SUMMARY:
{summary}

QUESTIONS:
{question_block}

ANSWERS (provide an answer for each question, prefixed with 'A: '):
"""

    def _get_qa_groundedness_evaluation_prompt(self, summary: str, questions_and_answers: str) -> str:
        """Prompt to evaluate if the answers are well-supported by the summary."""
        return f"""You are an expert evaluator. Given the original summary, a set of questions, and the answers generated *based only on that summary*, please evaluate how well each answer is supported by the information explicitly present in the summary.

Rate the overall support on a scale of 1-5:
1 = Very Poor (Answers are not supported by the summary, or fabricate information)
2 = Poor (Answers are mostly unsupported or significantly misinterpret the summary)
3 = Average (Some answers are supported, others are not or are partially supported)
4 = Good (Most answers are well-supported by the summary with minor discrepancies)
5 = Excellent (All answers are clearly and accurately supported by the summary)

ORIGINAL SUMMARY:
{summary}

QUESTIONS AND GENERATED ANSWERS:
{questions_and_answers}

EVALUATION:
Please provide your evaluation in the following format:
GROUNDEDNESS_SUPPORT: [score] - [brief overall explanation of why this score was given, highlighting any specific examples of good or poor support]
"""

    def evaluate_summary_groundedness_qa(self, summary: str) -> Dict[str, Any]:
        """
        Evaluate summary groundedness using a Q&A approach with LLM-as-a-Judge.
        """
        if not summary or not summary.strip():
            return {
                "llm_judge_groundedness_support": 0,
                "llm_judge_groundedness_questions": "Error: Empty summary",
                "llm_judge_groundedness_answers": "",
                "llm_judge_evaluation_text": "Error: Empty summary for Q&A groundedness",
                "llm_judge_metadata": {"error": "Empty summary provided for Q&A groundedness"}
            }

        try:
            # 1. Generate Questions
            qg_prompt = self._get_question_generation_prompt(summary)
            generated_questions_text, _, qg_metadata = self._call_judge_llm(qg_prompt)
            questions = [q.strip() for q in generated_questions_text.split('\\n') if q.strip()]
            if not questions:
                 return {
                    "llm_judge_groundedness_support": 0,
                    "llm_judge_groundedness_questions": "Error: Failed to generate questions from summary",
                    "llm_judge_groundedness_answers": "",
                    "llm_judge_evaluation_text": "Error: No questions generated for Q&A groundedness",
                    "llm_judge_metadata": {"error": "No questions generated", **qg_metadata}
                }


            # 2. Generate Answers based *only* on the summary
            ag_prompt = self._get_answer_generation_prompt(summary, questions)
            generated_answers_text, _, ag_metadata = self._call_judge_llm(ag_prompt)
            
            # 3. Evaluate Answer Groundedness
            eval_prompt = self._get_qa_groundedness_evaluation_prompt(summary, f"QUESTIONS:\\n{generated_questions_text}\\n\\nANSWERS:\\n{generated_answers_text}")
            evaluation_text, _, eval_metadata = self._call_judge_llm(eval_prompt)
            scores = self._parse_evaluation_scores(evaluation_text)

            # Combine metadata from all steps
            combined_metadata = {"qg_metadata": qg_metadata, "ag_metadata": ag_metadata, **eval_metadata}

            result = {
                "llm_judge_groundedness_questions": generated_questions_text,
                "llm_judge_groundedness_answers": generated_answers_text,
                "llm_judge_evaluation_text": evaluation_text,
                "llm_judge_metadata": combined_metadata,
                **scores # This should include llm_judge_groundedness_support
            }
            # Ensure the main score key is present, even if parsing failed to find it specifically
            if "llm_judge_groundedness_support" not in result:
                result["llm_judge_groundedness_support"] = 0


            return result

        except Exception as e:
            return {
                "llm_judge_groundedness_support": 0,
                "llm_judge_groundedness_questions": f"Error: {str(e)}",
                "llm_judge_groundedness_answers": "",
                "llm_judge_evaluation_text": f"Error during Q&A groundedness: {str(e)}",
                "llm_judge_metadata": {"error": str(e)}
            }

    def calculate(self, llm_response: str, task_type: str, **kwargs) -> Dict[str, Any]:
        """
        Generic calculate method to route to specific evaluation types.
        Args:
            llm_response: The response from the LLM.
            task_type: 'summarisation' or 'classification'.
            **kwargs: Additional arguments (e.g., 'transcript' for summarisation, 
                                            'input_text' for classification).
        Returns:
            A dictionary of calculated LLM-as-a-judge metrics.
        """
        if task_type == "summarisation":
            transcript = kwargs.get("transcript")
            return self.evaluate_summary(summary=llm_response, transcript=transcript)
        elif task_type == "classification":
            input_text = kwargs.get("input_text")
            return self.evaluate_classification(llm_response=llm_response, input_text=input_text)
        elif task_type == "summary_groundedness":
            return self.evaluate_summary_groundedness_qa(summary=llm_response)
        else:
            return {"error": f"Unsupported task_type for LLMJudgeEvaluator: {task_type}"}

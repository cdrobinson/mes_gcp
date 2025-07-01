# Metrics Documentation

This document describes the available evaluation metrics in the Model Evaluation Suite (MES) framework.

## Available Metrics

### 1. Transcript Quality Metric (`transcript_quality`)

**Use Case:** `transcription`  
**Description:** Evaluates the quality and format compliance of audio transcription outputs using native LLM metrics and format validation.

**Measured Aspects:**
- **Confidence Scores:** Uses log probabilities from the LLM response to assess confidence
- **Format Compliance:** Validates transcript structure including speaker labels and timestamps
- **Speaker Coverage:** Ensures proper speaker identification (Call_Agent, Customer)
- **Timestamp Coverage:** Checks for appropriate temporal markers

**Output Metrics:**
- `transcript_avg_log_probability` - Average log probability from the model
- `transcript_confidence` - Exponential of average log probability (confidence score)
- `transcript_timestamp_coverage` - Proportion of lines with timestamps
- `transcript_speaker_coverage` - Proportion of lines with speaker labels
- `transcript_format_compliance` - Overall format adherence score
- `transcript_required_speakers` - Presence of required speaker roles

### 2. Safety Metric (`safety`)

**Use Case:** `all`  
**Description:** Evaluates content safety using Google Cloud Platform's Model Armour service to detect harmful content, PII, and other safety concerns.

**Measured Aspects:**
- **CSAM Detection:** Identifies child safety and abuse material
- **PII Detection:** Detects personally identifiable information
- **Content Sanitisation:** Overall safety assessment
- **Data De-identification:** Checks for sensitive data transformation needs

**Output Metrics:**
- `safety_overall_flagged` - Whether any safety issues were detected
- `safety_invocation_success` - Whether the safety check completed successfully
- `safety_csam_flagged` - CSAM content detection flag
- `safety_csam_execution_success` - CSAM check execution status
- `safety_pii_flagged` - PII detection flag
- `safety_pii_findings_count` - Number of PII findings
- `safety_deidentify_flagged` - Whether de-identification is recommended

### 3. Vertex AI Evaluation Metric (`vertexai_evaluation`)

**Use Case:** `all`  
**Description:** Leverages Vertex AI's evaluation service for LLM-as-a-judge assessments using sophisticated evaluation criteria.

**Measured Aspects:**
- **Quality Assessment:** Overall response quality evaluation
- **Relevance:** How well the response addresses the input
- **Coherence:** Logical flow and consistency of the response
- **Groundedness:** Factual accuracy and adherence to source material

**Available Pointwise Metrics:**
- `vertexai_fluency`: Assesses the language quality and readability of the response.
- `vertexai_coherence`: Measures the logical flow and consistency of the response.
- `vertexai_safety`: Evaluates the safety of the response content.
- `vertexai_groundedness`: Checks if the response is factually accurate and supported by the provided context.
- `vertexai_instruction_following`: Assesses how well the model follows the instructions in the prompt.
- `vertexai_verbosity`: Measures the conciseness of the response.
- `vertexai_text_quality`: A general measure of the overall quality of the text.
- `vertexai_summarization_quality`: Specifically evaluates the quality of summaries.
- `vertexai_question_answering_quality`: Assesses the quality of answers to questions.

**Output Metrics:**
- The output will contain a column for each of the configured pointwise metrics (e.g., `vertexai_fluency_score`, `vertexai_groundedness_score`).
- Each metric will have a corresponding `_explanation` column with the rationale from the evaluator.

**Special Features:**
- Supports batch evaluation for improved efficiency
- Uses advanced LLM models for nuanced evaluation
- Provides detailed scoring rationale

## Metric Implementation

### Base Metric Class

All metrics inherit from `BaseMetric` which provides:
- Standard interface for metric computation
- Use case applicability checking
- Optional batch evaluation support
- Error handling and logging

### Adding Custom Metrics

To create a new metric:

1. Inherit from `BaseMetric`
2. Implement required methods:
   - `compute()` - Core metric computation
   - `get_description()` - Metric description
3. Optionally implement `batch_compute()` for batch evaluation
4. Register in `ExperimentRunner.__init__()`

### Batch vs Individual Evaluation

**Individual Evaluation:** Metrics are computed for each response independently during experiment execution.

**Batch Evaluation:** Metrics that support batch processing (like VertexAI Evaluation) compute scores for all responses together after individual processing completes, enabling more sophisticated cross-response analysis.

## Metric Naming Convention

Metrics follow a consistent naming pattern:
- `{metric_category}_{specific_measure}` (e.g., `transcript_confidence`)
- Boolean flags use `_flagged` suffix
- Counts use `_count` suffix  
- Success indicators use `_success` suffix
- Scores typically range from 0.0 to 1.0 unless otherwise specified

# Experiment Results Schema

This document describes the structure and fields of the experiment results DataFrame returned by the Model Evaluation Suite.

## Standard Fields

These fields are present in every experiment result regardless of the metrics used.

### Experiment Identification
- **`experiment_name`** (string) - Name of the experiment as defined in the configuration
- **`model_id`** (string) - Identifier of the LLM model used (e.g., "gemini-1.5-pro")
- **`use_case`** (string) - The use case category (e.g., "transcription", "summarisation")

### Input/Output Data
- **`audio_file`** (string) - GCS URI of the processed audio file
- **`response_text`** (string) - The complete text response from the LLM
- **`metadata`** (object) - Raw response metadata from the LLM service

### Token Usage
- **`input_tokens`** (integer) - Number of input tokens consumed
- **`output_tokens`** (integer) - Number of output tokens generated  
- **`total_tokens`** (integer) - Total token count (input + output)

### Processing Information
- **`processing_time`** (float) - Time taken to process this audio file (seconds)
- **`timestamp`** (string) - ISO timestamp when processing completed

### Error Handling
- **`error`** (string, optional) - Error message if processing failed

## Metric Score Fields

These fields are dynamically added based on the metrics configured for each experiment.

### Transcript Quality Metrics

Added when `transcript_quality` metric is enabled:

- **`transcript_avg_log_probability`** (float) - Average log probability from model output
- **`transcript_confidence`** (float) - Confidence score derived from log probabilities (0.0-1.0)
- **`transcript_timestamp_coverage`** (float) - Proportion of transcript lines with timestamps (0.0-1.0)
- **`transcript_speaker_coverage`** (float) - Proportion of transcript lines with speaker labels (0.0-1.0)
- **`transcript_format_compliance`** (float) - Overall format adherence score (0.0-1.0)
- **`transcript_required_speakers`** (float) - Whether required speakers are present (0.0 or 1.0)

### Summarisation Quality Metrics

Added when `summarization_quality` metric is enabled:

- **`summarisation_avg_log_probability`** (float) - Average log probability from model output
- **`summarisation_confidence`** (float) - Confidence score derived from log probabilities (0.0-1.0)
- **`summarisation_section_coverage`** (float) - Proportion of expected sections present (0.0-1.0)
- **`summarisation_section_length_compliance`** (float) - Adherence to 100-200 word section requirement (0.0-1.0)
- **`summarisation_format_compliance`** (float) - Plain text format compliance score (0.0-1.0)

### Safety Metrics

Added when `safety` metric is enabled:

- **`safety_overall_flagged`** (float) - Whether any safety issues detected (0.0 or 1.0)
- **`safety_invocation_success`** (float) - Whether safety check completed successfully (0.0 or 1.0)
- **`safety_csam_flagged`** (float) - CSAM content detection result (0.0 or 1.0)
- **`safety_csam_execution_success`** (float) - CSAM check execution status (0.0 or 1.0)
- **`safety_pii_flagged`** (float) - PII detection result (0.0 or 1.0)
- **`safety_pii_findings_count`** (float) - Number of PII findings detected
- **`safety_deidentify_flagged`** (float) - Whether de-identification recommended (0.0 or 1.0)

### Vertex AI Evaluation Metrics

Added when `vertexai_evaluation` metric is enabled:

- **`vertexai_quality_score`** (float) - Overall quality assessment (0.0-1.0)
- **`vertexai_relevance_score`** (float) - Response relevance score (0.0-1.0)
- **`vertexai_coherence_score`** (float) - Logical consistency score (0.0-1.0)
- **`vertexai_groundedness_score`** (float) - Factual accuracy score (0.0-1.0)

### Error Tracking Fields

For each metric, additional error tracking fields may be present:

- **`{metric_name}_error`** (float) - Set to 1.0 if metric computation failed
- **`{metric_name}_batch_error`** (float) - Set to 1.0 if batch evaluation failed

## Data Types and Ranges

### Score Ranges
- **Binary indicators:** 0.0 (false/absent) or 1.0 (true/present)
- **Proportions/Coverage:** 0.0 to 1.0 representing percentages
- **Quality scores:** 0.0 to 1.0 where higher is better
- **Safety scores:** 0.0 to 1.0 where lower typically indicates safer content
- **Counts:** Non-negative integers converted to float

### Missing Values
- Missing or failed metric computations result in NaN values
- Error fields are set to 1.0 when computation fails
- Standard fields should never be missing unless processing completely failed

## Example Result Record

```python
{
    # Standard fields
    'experiment_name': 'transcription_baseline',
    'model_id': 'gemini-1.5-pro',
    'use_case': 'transcription',
    'audio_file': 'gs://bucket/calls/claim_001.wav',
    'response_text': 'Call_Agent: How can I help you today?\nCustomer: I need to file a claim...',
    'input_tokens': 1250,
    'output_tokens': 890,
    'total_tokens': 2140,
    'processing_time': 12.34,
    'timestamp': '2025-06-26T10:30:45',
    
    # Transcript quality metrics
    'transcript_confidence': 0.87,
    'transcript_format_compliance': 0.95,
    'transcript_speaker_coverage': 1.0,
    
    # Summarization quality metrics (when use_case is 'summarization')
    'summarization_confidence': 0.92,
    'summarization_section_coverage': 0.78,
    'summarization_format_compliance': 0.95,
    'summarization_required_sections': 1.0,
    
    # Safety metrics  
    'safety_overall_flagged': 0.0,
    'safety_pii_flagged': 0.0,
    'safety_pii_findings_count': 0.0,
    
    # Vertex AI evaluation
    'vertexai_quality_score': 0.82,
    'vertexai_relevance_score': 0.91
}
```

## Usage Notes

### DataFrame Operations
- All numeric fields support standard pandas aggregation operations
- Group by `experiment_name` for cross-experiment comparison
- Group by `model_id` for model performance analysis
- Filter on error fields to identify processing issues

### Analysis Patterns
```python
# Calculate average metrics per experiment
experiment_summary = df.groupby('experiment_name').agg({
    'transcript_confidence': 'mean',
    'safety_overall_flagged': 'mean',
    'processing_time': ['mean', 'sum']
})

# Find best performing model
model_performance = df.groupby('model_id')[metric_columns].mean()
best_model = model_performance.mean(axis=1).idxmax()
```

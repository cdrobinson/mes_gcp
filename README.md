# Model Evaluation Suite (MES)

A framework for evaluating Large Language Models (LLMs) on audio transcription, summarisation and analysis tasks using Google Cloud Platform services.

## Quick Start

### 1. Configure Experiments

Create a YAML configuration file with your experiment settings. See `config/sample_experiments.yaml` for reference:

```yaml
project: "your-gcp-project-id"
location: "global"
bucket: "your-audio-bucket"

experiments:
  - name: "transcription_baseline"
    use_case: "transcription"
    model_id: "gemini-1.5-pro"
    prompt_id: "prompt-v1"
    metrics:
      - "transcript_quality"
      - "safety"
```

### 2. Run Experiments

Use the `ExperimentRunner` to execute your experiments:

```python
from orchestrator.experiment_runner import ExperimentRunner

# Initialise with your config
runner = ExperimentRunner('config/your_config.yaml')

# Run all experiments
results = runner.run_experiments()

# Or run specific experiments
results = runner.run_experiments(['transcription_baseline'])
```

Alternatively, use the Jupyter notebook in `notebooks/run_experiments.ipynb` for interactive analysis.

## Extending the Framework

### Adding New Clients

Create new clients by implementing the appropriate interface pattern. Existing clients include:
- `GCSClient` - Google Cloud Storage
- `GeminiClient` - Gemini LLM models
- `BigQueryClient` - BigQuery data warehouse
- `VertexAIClient` - Vertex AI services

### Adding New Metrics

Extend the framework by creating custom metrics that inherit from `BaseMetric`:

```python
from metrics.base_metric import BaseMetric

class CustomMetric(BaseMetric):
    def __init__(self):
        super().__init__("custom_metric", "transcription")
    
    def compute(self, response: str, metadata: Dict[str, Any]) -> Dict[str, float]:
        # Your metric logic here
        return {"custom_score": 0.85}
    
    def get_description(self) -> str:
        return "Description of what this metric measures"
```

Register your metric in the `ExperimentRunner.__init__()` method to make it available for experiments.

### Available Metrics

- **transcript_quality** - Evaluates transcription accuracy and format compliance
- **summarisation_quality** - Evaluates structured summary quality and format compliance
- **safety** - Content safety evaluation using GCP Model Armour
- **vertexai_evaluation** - LLM-as-a-judge evaluations via Vertex AI

## Project Structure

- `src/orchestrator/` - Experiment execution and coordination
- `src/clients/` - GCP service integrations
- `src/metrics/` - Evaluation metric implementations
- `config/` - Experiment configuration files
- `notebooks/` - Interactive analysis and visualisation
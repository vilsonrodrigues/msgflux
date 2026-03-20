# nn.Predictor

The `nn.Predictor` module provides a unified interface for wrapping machine learning models (classifiers, regressors, object detectors) into msgFlux workflows. It normalizes inputs/outputs and provides integration with `Message` objects.

All code examples use the recommended import pattern:

```python
import msgflux as mf
import msgflux.nn as nn
```

## Quick Start

### AutoParams Initialization (Recommended)

Defines specialized predictors with clear semantics.

```python
import msgflux as mf
import msgflux.nn as nn

class SentimentPredictor(nn.Predictor):
    """Predicts sentiment (positive/negative) from text."""
    response_mode = "plain_response"

# Load your ML model (scikit-learn, pytorch, etc.)
sentiment_model = load_pretrained_model("sentiment_v1.pkl")

# Create predictor instance
predictor = SentimentPredictor(
    model=sentiment_model,
    config={"threshold": 0.7}
)

# Predict
sentiment = predictor("This product is amazing!")
print(sentiment)
```

### Traditional Initialization

```python
predictor = nn.Predictor(
    model=sentiment_model,
    config={"threshold": 0.5}
)
```

---

## Use Cases

### 1. Text Classification

Wrap NLP models for classification tasks.

```python
class SpamClassifier(nn.Predictor):
    """Classifies emails as spam or not."""
    response_mode = "plain_response"

spam_detector = SpamClassifier(
    model=load_model("spam_classifier.pkl"),
    config={"threshold": 0.8}
)

is_spam = spam_detector("Congratulations! You've won $1,000,000!")
```

### 2. Computer Vision

Wrap vision models for detection or classification.

```python
class ObjectDetector(nn.Predictor):
    """Detects objects in images."""
    response_mode = "plain_response"

detector = ObjectDetector(
    model=load_model("yolo_v8.pt"),
    config={"confidence": 0.5}
)

objects = detector("image.jpg")
```

### 3. Regression & Time Series

Wrap regression models for forecasting.

```python
class PricePredictor(nn.Predictor):
    """Predicts prices based on features."""
    response_mode = "plain_response"

predictor = PricePredictor(model=load_model("price_regressor.pkl"))

price = predictor({"sqft": 2000, "bedrooms": 3})
```

---

## Advanced Configuration

### Message Field Mapping

Map specific fields from a `Message` object to your model inputs.

```python
class StructuredPredictor(nn.Predictor):
    """Predictor that reads from message features."""
    response_mode = "message"

predictor = StructuredPredictor(
    model=model,
    message_fields={
        "task_inputs": "features.dense_vector"
    }
)

msg = mf.Message()
msg.set("features.dense_vector", [0.1, 0.5, 0.9])

result_msg = predictor(msg)
prediction = result_msg.get("predictor.result")
```

### Response Templates

Format the prediction output using Jinja templates.

```python
class FormattedPredictor(nn.Predictor):
    """Predictor with formatted string output."""
    response_mode = "plain_response"

predictor = FormattedPredictor(
    model=model,
    response_template="""
    Result: {{ prediction }}
    Confidence: {{ confidence }}%
    """
)

print(predictor(input_data))
```

---

## Creating Predictor Hierarchies

Build a hierarchy of predictors to share configuration logic.

```python
# Base predictor for binary classification
class BaseClassifier(nn.Predictor):
    """Base class for classifiers."""
    response_mode = "plain_response"

class FraudDetector(BaseClassifier):
    """Specialized fraud detector."""

class ChurnPredictor(BaseClassifier):
    """Specialized churn predictor."""

# Instantiate
fraud_model = FraudDetector(
    model=fraud_model,
    config={"threshold": 0.9} # Strict
)

churn_model = ChurnPredictor(
    model=churn_model,
    config={"threshold": 0.5} # Balanced
)
```

---

## Integration with Agents

Predictors are excellent tools for Agents. They allow agents to offload specialized tasks to ML models.

```python
# Define predictor
analyzer = SentimentPredictor(model=sentiment_model)

# Define as tool function
def analyze_sentiment(text: str) -> str:
    """Analyze sentiment of the customer message."""
    return analyzer(text)

# Agent with access to the predictor
class SupportAgent(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4")
    tools = [analyze_sentiment]

agent = SupportAgent()
response = agent("Customer says: 'I love your service but hate the app.'")
```

---

## Batch & Async

Efficiently process batches of data using `aforward`.

```python
import asyncio

async def predict_batch(inputs):
    tasks = [predictor.aforward(inp) for inp in inputs]
    return await asyncio.gather(*tasks)

inputs = ["text 1", "text 2", "text 3"]
results = asyncio.run(predict_batch(inputs))
```

---

## Best Practices

1.  **Task Specificity**: Create specific predictor classes (`FraudDetector`, `SpamClassifier`) instead of generic `Predictor` instances.
2.  **Input Validation**: Use standard Python validation functions before passing data to the predictor if your model is sensitive to input types.
3.  **Confidence Thresholds**: Expose thresholds in `config` to allow easy tuning without changing code.

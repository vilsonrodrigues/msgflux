# Predictor: ML Model Wrapper Module

The `Predictor` module provides a unified interface for wrapping machine learning models (classifiers, regressors, detectors, segmenters) into msgflux workflows.

**All code examples use the recommended import pattern:**

```python
import msgflux as mf
```

## Quick Start

### Traditional Initialization

```python
import msgflux as mf
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Train a scikit-learn model
X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y_train = np.array([0, 0, 1, 1])

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Wrap in msgflux BaseModel (example - you'd create custom wrapper)
wrapped_model = CustomSklearnModel(rf_model)

# Traditional predictor initialization
predictor = mf.nn.Predictor(
    model=wrapped_model,
    config={"return_probabilities": True}
)

# Make predictions
result = predictor([[2, 3]])
print(result)
```

### AutoParams Initialization (Recommended)

**This is the preferred and recommended way to define predictors in msgflux.**

```python
import msgflux as mf

class SentimentPredictor(mf.nn.Predictor):
    """Predicts sentiment (positive/negative) from text."""

    # AutoParams automatically uses class name as 'name'
    response_mode = "plain_response"

# Create or load your ML model
sentiment_model = load_sentiment_model()

# Create predictor with defaults
predictor = SentimentPredictor(
    model=sentiment_model,
    config={"threshold": 0.5}
)

# Predict
text = "This product is amazing!"
sentiment = predictor(text)
print(sentiment)  # "positive"

# Create variant with different threshold
strict_predictor = SentimentPredictor(
    model=sentiment_model,
    config={"threshold": 0.7}  # Higher confidence required
)
```

## Why Use AutoParams?

1. **Model-Specific Predictors**: Create specialized predictors for different ML tasks
2. **Reusable Configuration**: Share model configuration across predictions
3. **Clear Purpose**: Class name and docstring document predictor's task
4. **Easy Variants**: Create predictor variations with different thresholds/parameters
5. **Better Organization**: Group predictors by ML task or domain

## Use Cases

### 1. Text Classification

```python
import msgflux as mf

class SpamClassifier(mf.nn.Predictor):
    """Classifies emails as spam or not spam."""

    response_mode = "plain_response"

class TopicClassifier(mf.nn.Predictor):
    """Classifies documents into topic categories."""

    response_mode = "plain_response"

# Load pre-trained models
spam_model = load_model("spam_classifier.pkl")
topic_model = load_model("topic_classifier.pkl")

# Create predictors
spam_detector = SpamClassifier(
    model=spam_model,
    config={"threshold": 0.8}  # High confidence for spam
)

topic_detector = TopicClassifier(
    model=topic_model,
    config={"top_k": 3}  # Return top 3 topics
)

# Use
email_text = "Congratulations! You've won $1,000,000!"
is_spam = spam_detector(email_text)

article_text = "Recent advances in quantum computing have..."
topics = topic_detector(article_text)
```

### 2. Image Classification

```python
import msgflux as mf

class ObjectDetector(mf.nn.Predictor):
    """Detects objects in images."""

    response_mode = "plain_response"

class ImageClassifier(mf.nn.Predictor):
    """Classifies images into categories."""

    response_mode = "plain_response"

# Load computer vision models
detector_model = load_model("yolo_detector.pt")
classifier_model = load_model("resnet_classifier.pt")

# Create predictors
detector = ObjectDetector(
    model=detector_model,
    config={"confidence": 0.5, "iou_threshold": 0.45}
)

classifier = ImageClassifier(
    model=classifier_model,
    config={"top_k": 5}
)

# Use
image_path = "photo.jpg"
objects = detector(image_path)
categories = classifier(image_path)
```

### 3. Regression

```python
import msgflux as mf

class PricePredictor(mf.nn.Predictor):
    """Predicts house prices based on features."""

    response_mode = "plain_response"

class DemandForecaster(mf.nn.Predictor):
    """Forecasts product demand."""

    response_mode = "plain_response"

# Load regression models
price_model = load_model("price_regressor.pkl")
demand_model = load_model("demand_forecaster.pkl")

# Create predictors
price_predictor = PricePredictor(model=price_model)
demand_forecaster = DemandForecaster(model=demand_model)

# Use
house_features = {"sqft": 2000, "bedrooms": 3, "location": "downtown"}
predicted_price = price_predictor(house_features)

product_features = {"day_of_week": 5, "promotions": True, "weather": "sunny"}
predicted_demand = demand_forecaster(product_features)
```

### 4. Time Series Analysis

```python
import msgflux as mf

class StockPredictor(mf.nn.Predictor):
    """Predicts stock price movements."""

    response_mode = "plain_response"

class AnomalyDetector(mf.nn.Predictor):
    """Detects anomalies in time series data."""

    response_mode = "plain_response"

# Load time series models
stock_model = load_model("lstm_stock.pt")
anomaly_model = load_model("isolation_forest.pkl")

# Create predictors
stock_predictor = StockPredictor(
    model=stock_model,
    config={"horizon": 5}  # Predict 5 days ahead
)

anomaly_detector = AnomalyDetector(
    model=anomaly_model,
    config={"contamination": 0.1}
)

# Use
historical_prices = [100, 102, 101, 103, 105]
future_prices = stock_predictor(historical_prices)

sensor_data = get_sensor_readings()
anomalies = anomaly_detector(sensor_data)
```

## Message Field Mapping

Use Message objects for structured input/output:

```python
import msgflux as mf

class StructuredPredictor(mf.nn.Predictor):
    """Predictor that processes structured message inputs."""

    response_mode = "message"

model = load_model("classifier.pkl")

predictor = StructuredPredictor(
    model=model,
    message_fields={
        "task_inputs": "features.data"
    }
)

# Create message with features
msg = mf.Message()
msg.set("features.data", [1.5, 2.3, 0.8, 4.1])
msg.set("metadata.id", "prediction_123")

# Predict and get result in message
result_msg = predictor(msg)
prediction = result_msg.get("predictor.result")
metadata_id = result_msg.get("metadata.id")

print(f"Prediction {metadata_id}: {prediction}")
```

## Response Templates

Format predictions using Jinja templates:

```python
import msgflux as mf

class FormattedPredictor(mf.nn.Predictor):
    """Predictor with custom output formatting."""

    response_mode = "plain_response"

model = load_model("sentiment_model.pkl")

predictor = FormattedPredictor(
    model=model,
    response_template="""
    Sentiment Analysis Result:
    - Sentiment: {{ prediction }}
    - Confidence: {{ confidence }}%
    - Model: {{ model_name }}
    """
)

formatted_result = predictor("This is great!")
print(formatted_result)
```

## Creating Predictor Hierarchies

Build specialized predictors through inheritance:

```python
import msgflux as mf

# Base predictor for all classifiers
class BaseClassifier(mf.nn.Predictor):
    """Base predictor for classification tasks."""

    response_mode = "plain_response"

# Binary classifier
class BinaryClassifier(BaseClassifier):
    """Binary classification (yes/no, spam/ham, etc.)."""

    # Inherits response_mode from BaseClassifier

# Multi-class classifier
class MultiClassClassifier(BaseClassifier):
    """Multi-class classification."""

    # Different configuration for multi-class

# Load models
binary_model = load_model("binary_classifier.pkl")
multiclass_model = load_model("multiclass_classifier.pkl")

# Create instances
binary = BinaryClassifier(
    model=binary_model,
    config={"threshold": 0.5}
)

multiclass = MultiClassClassifier(
    model=multiclass_model,
    config={"top_k": 1}  # Return top class only
)

# Use appropriately
is_fraudulent = binary("Transaction details...")
document_category = multiclass("Document text...")
```

## Integration with Agents

Predictors can be used as tools for agents:

```python
import msgflux as mf

# Define predictor
class SentimentAnalyzer(mf.nn.Predictor):
    """Analyzes sentiment of text."""

    response_mode = "plain_response"

sentiment_model = load_model("sentiment.pkl")

analyzer = SentimentAnalyzer(model=sentiment_model)

# Define predictor as a tool function
def analyze_sentiment(text: str) -> str:
    """Analyze the sentiment of the given text."""
    result = analyzer(text)
    return result

# Create agent with predictor tool
class CustomerSupportAgent(mf.nn.Agent):
    """Customer support agent with sentiment analysis."""

    temperature = 0.7
    max_tokens = 2000

model = mf.Model.chat_completion("openai/gpt-4")

support_agent = CustomerSupportAgent(
    model=model,
    tools=[analyze_sentiment]
)

# Agent can now use predictor to analyze customer sentiment
response = support_agent("""
Customer message: "I'm very disappointed with your service."
Analyze sentiment and respond appropriately.
""")
print(response)
```

## Batch Prediction

Make predictions on multiple inputs efficiently:

```python
import msgflux as mf
import asyncio

class BatchPredictor(mf.nn.Predictor):
    """Predictor for batch processing."""

    response_mode = "plain_response"

model = load_model("classifier.pkl")

predictor = BatchPredictor(model=model)

async def predict_batch(inputs):
    """Predict on multiple inputs concurrently."""
    tasks = [predictor.aforward(inp) for inp in inputs]
    return await asyncio.gather(*tasks)

# Batch predict
inputs = ["text 1", "text 2", "text 3", "text 4"]
results = asyncio.run(predict_batch(inputs))

for inp, result in zip(inputs, results):
    print(f"{inp}: {result}")
```

## Model Gateway Support

Use multiple models with fallback:

```python
import msgflux as mf

class EnsemblePredictor(mf.nn.Predictor):
    """Predictor using multiple models for ensemble."""

    response_mode = "plain_response"

# Create model gateway with multiple models
gateway = mf.ModelGateway(
    models={
        "primary": load_model("model_v2.pkl"),
        "fallback": load_model("model_v1.pkl")
    }
)

predictor = EnsemblePredictor(
    model=gateway,
    message_fields={
        "model_preference": "config.preferred_model"
    }
)

# Predict with model preference
msg = mf.Message()
msg.set("features", [1, 2, 3])
msg.set("config.preferred_model", "primary")

result = predictor(msg)
```

## Configuration Options

### Complete Parameter Reference

```python
import msgflux as mf

class FullyConfiguredPredictor(mf.nn.Predictor):
    """Predictor with all configuration options."""

    # Response behavior
    response_mode = "plain_response"  # or "message"

model = load_model("model.pkl")

predictor = FullyConfiguredPredictor(
    model=model,                         # Required: ML model
    message_fields={                     # Optional: Message field mapping
        "task_inputs": "data.features",
        "model_preference": "config.model"
    },
    response_template="...",             # Optional: Jinja template
    config={                             # Optional: model-specific config
        # All parameters passed directly to model
        "threshold": 0.5,
        "top_k": 3,
        "temperature": 0.7
    },
    name="custom_predictor"              # Optional: custom name
)
```

## Best Practices

### 1. Use Task-Specific Predictors

```python
# Good - Clear, specialized predictors
class FraudDetector(mf.nn.Predictor):
    """Detects fraudulent transactions."""
    response_mode = "plain_response"

class ChurnPredictor(mf.nn.Predictor):
    """Predicts customer churn risk."""
    response_mode = "plain_response"

class RecommendationEngine(mf.nn.Predictor):
    """Generates product recommendations."""
    response_mode = "plain_response"
```

### 2. Validate Inputs

```python
class ValidatedPredictor(mf.nn.Predictor):
    """Predictor with input validation."""
    response_mode = "plain_response"

def validate_features(features):
    """Validate features before prediction."""
    if not isinstance(features, list):
        raise ValueError("Features must be a list")
    if len(features) != 10:
        raise ValueError("Expected 10 features")
    return features

# Use validation in preprocessing
predictor = ValidatedPredictor(model=model)

try:
    result = predictor(validate_features(user_input))
except ValueError as e:
    print(f"Invalid input: {e}")
```

### 3. Handle Uncertainty

```python
class ConfidencePredictor(mf.nn.Predictor):
    """Predictor that returns confidence scores."""
    response_mode = "plain_response"

predictor = ConfidencePredictor(
    model=model,
    config={"return_confidence": True}
)

result = predictor(input_data)

if result["confidence"] < 0.7:
    print("Low confidence - manual review recommended")
else:
    print(f"Prediction: {result['prediction']} (confidence: {result['confidence']})")
```

## Migration Guide

### From Traditional to AutoParams

**Before (Traditional):**
```python
predictor = mf.nn.Predictor(
    model=ml_model,
    config={"threshold": 0.5, "top_k": 3}
)
```

**After (AutoParams - Recommended):**
```python
class MyPredictor(mf.nn.Predictor):
    """Predictor for my specific ML task."""
    response_mode = "plain_response"

predictor = MyPredictor(
    model=ml_model,
    config={"threshold": 0.5, "top_k": 3}
)
```

## Summary

- **Use AutoParams** for defining predictors - create specialized predictors for different ML tasks
- **Traditional initialization** works for quick, one-off predictions
- Supports **any ML model** (classifiers, regressors, detectors, segmenters)
- Works with **scikit-learn**, **PyTorch**, **TensorFlow**, and custom models
- Configure via **message_fields**, **response_template**, and **config** options
- Commonly integrated with **Agents** as tools
- **Async support** for batch predictions
- **Model Gateway** support for ensemble and fallback

The Predictor module provides a unified interface for ML models - use AutoParams to organize predictors by task and domain.

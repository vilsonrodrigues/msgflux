# Model Gateway

When using a `ModelGateway` with multiple models, you can specify which model to use via `model_preference`:

```python
# pip install msgflux[openai]
import msgflux as mf
import msgflux.nn as nn

# mf.set_envs(OPENAI_API_KEY="...")

# Create gateway with named model deployments
gateway = mf.ModelGateway([
    {"model_name": "low_cost", "model": mf.Model.chat_completion("openai/gpt-4.1-mini")},
    {"model_name": "high_quality", "model": mf.Model.chat_completion("openai/gpt-5.2")},
])

agent = nn.Agent("agent", gateway)

# Use specific model for simple tasks
response = agent("Tell me a joke", model_preference="low_cost")

# Use better model for complex tasks
response = agent("Analyze this contract...", model_preference="high_quality")
```

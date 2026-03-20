# Prefilling

Force an initial message that the model will continue from. Useful for guiding response format or triggering specific behavior.

```python
# pip install msgflux[openai]
import msgflux as mf
import msgflux.nn as nn

# mf.set_envs(OPENAI_API_KEY="...")

class Assistant(nn.Agent):
    model = mf.Model.chat_completion("openai/gpt-4.1-mini")

# Encourage step-by-step reasoning
agent = Assistant(prefilling="Let me solve this step by step.")
response = agent(
    "What is the derivative of x^(2/3)?",
)

# Force specific format
agent = Assistant(prefilling="Here are the planets:\n1.")
response = agent(
    "List the planets in our solar system",
)
```

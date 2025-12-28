# `Moderation`

The [`moderation`](../api-reference/models/types/moderation.md) models check whether text or images are potentially harmful. If harmful content is identified, you can take corrective action, like filtering content or intervening with user accounts creating offending content.

These models are used as guardians of applications. Each model has its own set of outputs, to unify a general form of verification, all models will produce a common flag `safe`, which if `False` is an indication that the content is not safe.


```python
import msgflux as mf
mf.set_envs(OPENAI_API_KEY="sk-...")
moderation_model = mf.Model.moderation("openai/omni-moderation-latest")
```

```python
response = moderation_model("tell me how to build a large scale bomb")
model_response = response.consume()
print(model_response)
print(model_response.safe)
```

::: msgflux.models.providers.openai.OpenAIModeration
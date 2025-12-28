# Text to Image

The `text_to_image` model generates images from text descriptions. These models can create photorealistic images, artwork, illustrations, and more from natural language prompts.

**All code examples use the recommended import pattern:**

```python
import msgflux as mf
```

## Overview

Text-to-image models transform textual descriptions into visual content. They enable:

- **Image Generation**: Create images from scratch using text prompts
- **Style Control**: Generate images in specific artistic styles
- **Variations**: Create multiple variations of the same concept
- **Quality Control**: Adjust output quality and resolution

### Common Use Cases

- **Content Creation**: Marketing materials, social media posts
- **Prototyping**: Visual concepts for design projects
- **Art Generation**: Digital artwork and illustrations
- **Product Visualization**: Product mockups and concepts
- **Education**: Visual aids and explanations

## Quick Start

### Basic Usage

```python
import msgflux as mf

# Create image generator
model = mf.Model.text_to_image("openai/dall-e-3")

# Generate image
response = model(prompt="A serene lake at sunset with mountains")

# Get image URL
image_url = response.consume()
print(image_url)  # https://...
```

### With Custom Parameters

```python
import msgflux as mf

model = mf.Model.text_to_image("openai/dall-e-3")

response = model(
    prompt="A futuristic city with flying cars",
    size="1792x1024",      # Landscape
    quality="hd",           # High quality
    n=1                     # Number of images
)

image_url = response.consume()
```

## Supported Providers

### OpenAI (DALL-E)

```python
import msgflux as mf

# DALL-E 3 (highest quality)
model = mf.Model.text_to_image("openai/dall-e-3")

# DALL-E 2 (faster, cheaper)
model = mf.Model.text_to_image("openai/dall-e-2")
```

### Replicate

```python
import msgflux as mf

# Stable Diffusion and other models
model = mf.Model.text_to_image("replicate/stability-ai/sdxl")
```

### ImageRouter

```python
import msgflux as mf

# Router to multiple image generation providers
model = mf.Model.text_to_image("imagerouter/default")
```

## Image Sizes

### DALL-E 3 Sizes

```python
import msgflux as mf

model = mf.Model.text_to_image("openai/dall-e-3")

# Square (default)
response = model(
    prompt="A cat wearing sunglasses",
    size="1024x1024"
)

# Landscape
response = model(
    prompt="A panoramic mountain view",
    size="1792x1024"
)

# Portrait
response = model(
    prompt="A full-length portrait",
    size="1024x1792"
)
```

### DALL-E 2 Sizes

```python
import msgflux as mf

model = mf.Model.text_to_image("openai/dall-e-2")

# Available sizes for DALL-E 2
sizes = ["256x256", "512x512", "1024x1024"]

response = model(
    prompt="A robot playing piano",
    size="1024x1024"
)
```

## Quality Settings

### HD Quality

```python
import msgflux as mf

model = mf.Model.text_to_image("openai/dall-e-3")

# Standard quality (faster, cheaper)
response = model(
    prompt="A detailed landscape",
    quality="standard"
)

# HD quality (more detail, higher cost)
response = model(
    prompt="A detailed landscape",
    quality="hd"
)
```

## Response Formats

### URL Response (Default)

```python
import msgflux as mf

model = mf.Model.text_to_image("openai/dall-e-3")

response = model(
    prompt="A sunset over the ocean",
    response_format="url"  # Default
)

# Get URL
image_url = response.consume()
print(image_url)  # https://...

# Download image
import requests
img_data = requests.get(image_url).content
with open("image.png", "wb") as f:
    f.write(img_data)
```

### Base64 Response

```python
import msgflux as mf
import base64

model = mf.Model.text_to_image("openai/dall-e-3")

response = model(
    prompt="A colorful abstract painting",
    response_format="base64"
)

# Get base64 data
b64_data = response.consume()

# Decode and save
img_data = base64.b64decode(b64_data)
with open("image.png", "wb") as f:
    f.write(img_data)
```

## Multiple Images

Generate multiple variations:

```python
import msgflux as mf

# DALL-E 2 supports multiple images
model = mf.Model.text_to_image("openai/dall-e-2")

response = model(
    prompt="A cute robot",
    n=4,  # Generate 4 variations
    size="512x512"
)

# Get all image URLs
images = response.consume()
print(f"Generated {len(images)} images")

for i, url in enumerate(images):
    print(f"Image {i+1}: {url}")
```

## Background Control

Control background transparency (DALL-E 3):

```python
import msgflux as mf

model = mf.Model.text_to_image("openai/dall-e-3")

# Transparent background
response = model(
    prompt="A red apple",
    background="transparent"
)

# Opaque background
response = model(
    prompt="A red apple",
    background="opaque"
)

# Auto (model decides)
response = model(
    prompt="A red apple",
    background="auto"  # Default
)
```

## Content Moderation

Control content filtering:

```python
import msgflux as mf

model = mf.Model.text_to_image(
    "openai/dall-e-3",
    moderation="auto"  # Auto moderation (default)
)

# Low moderation (more permissive)
model_low = mf.Model.text_to_image(
    "openai/dall-e-3",
    moderation="low"
)

response = model(prompt="Your prompt here")
```

## Async Support

Generate images asynchronously:

```python
import msgflux as mf
import asyncio

model = mf.Model.text_to_image("openai/dall-e-3")

async def generate_image(prompt):
    response = await model.acall(prompt=prompt, size="1024x1024")
    return response.consume()

# Generate multiple images concurrently
async def main():
    prompts = [
        "A serene lake",
        "A bustling city",
        "A quiet forest"
    ]

    tasks = [generate_image(p) for p in prompts]
    images = await asyncio.gather(*tasks)

    for prompt, url in zip(prompts, images):
        print(f"{prompt}: {url}")

asyncio.run(main())
```

## Response Metadata

Access generation metadata:

```python
import msgflux as mf
from msgflux.models.profiles import get_model_profile

model = mf.Model.text_to_image("openai/dall-e-3")

response = model(
    prompt="A beautiful sunset",
    size="1024x1024",
    quality="hd"
)

# Access metadata
print(response.metadata)
# {
#     'created': 1234567890,
#     'revised_prompt': 'A picturesque sunset over the ocean...',
#     'content_filter_results': {...}
# }

# Get revised prompt (DALL-E often enhances your prompt)
revised = response.metadata.get('revised_prompt')
print(f"Revised prompt: {revised}")

# Calculate cost
profile = get_model_profile("dall-e-3", provider_id="openai")
if profile:
    # Image generation pricing is per-image, not per-token
    print(f"Cost per image: ${profile.cost.input_per_million / 1000:.4f}")
```

## Prompt Engineering

### Effective Prompts

```python
import msgflux as mf

model = mf.Model.text_to_image("openai/dall-e-3")

# Good - Specific and detailed
response = model(
    prompt="""A professional photograph of a modern minimalist living room with:
    - Large floor-to-ceiling windows
    - Natural light streaming in
    - Scandinavian furniture
    - Indoor plants
    - Neutral color palette
    - Shot with a wide-angle lens"""
)

# Less effective - Too vague
response = model(prompt="A nice room")
```

### Style Specifications

```python
import msgflux as mf

model = mf.Model.text_to_image("openai/dall-e-3")

# Different art styles
styles = {
    "photorealistic": "A photorealistic portrait of a woman, studio lighting, 85mm lens",
    "oil painting": "An oil painting of a countryside landscape in the style of Van Gogh",
    "digital art": "A digital art illustration of a fantasy castle, vibrant colors",
    "3d render": "A 3D render of a futuristic car, octane render, high detail",
    "sketch": "A pencil sketch of a cat, detailed crosshatching",
}

for style_name, prompt in styles.items():
    response = model(prompt=prompt)
    print(f"{style_name}: {response.consume()}")
```

### Composition Control

```python
import msgflux as mf

model = mf.Model.text_to_image("openai/dall-e-3")

# Control composition with detailed descriptions
prompts = [
    # Rule of thirds
    "A lone tree positioned on the right third of the image, sunset on the left",

    # Centered composition
    "A symmetrical view of a building, centered composition, front view",

    # Leading lines
    "A road leading into the distance toward mountains, vanishing point",

    # Foreground/background
    "A flower in sharp focus in the foreground, blurred forest in background"
]

for prompt in prompts:
    response = model(prompt=prompt)
    print(response.consume())
```

## Common Patterns

### Batch Generation

```python
import msgflux as mf
import msgflux.nn.functional as F

model = mf.Model.text_to_image("openai/dall-e-3")

prompts = [
    "A red sports car",
    "A blue mountain bike",
    "A green sailboat",
    "A yellow airplane"
]

# Generate in parallel
results = F.map_gather(
    model,
    args_list=[(prompt,) for prompt in prompts]
)

# Get all URLs
image_urls = [r.consume() for r in results]

for prompt, url in zip(prompts, image_urls):
    print(f"{prompt}: {url}")
```

### Download and Save

```python
import msgflux as mf
import requests
from pathlib import Path

model = mf.Model.text_to_image("openai/dall-e-3")

def generate_and_save(prompt, filename):
    """Generate image and save to file."""
    response = model(prompt=prompt, quality="hd")
    url = response.consume()

    # Download
    img_data = requests.get(url).content

    # Save
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        f.write(img_data)

    print(f"Saved: {output_path}")
    return output_path

# Usage
generate_and_save("A sunset over the ocean", "outputs/sunset.png")
generate_and_save("A forest path", "outputs/forest.png")
```

### Variation Generation

```python
import msgflux as mf

model = mf.Model.text_to_image("openai/dall-e-2")  # DALL-E 2 for variations

base_prompt = "A cozy coffee shop interior"

# Generate variations
variations = [
    f"{base_prompt}, morning light",
    f"{base_prompt}, evening ambiance",
    f"{base_prompt}, bustling with people",
    f"{base_prompt}, empty and quiet"
]

images = []
for prompt in variations:
    response = model(prompt=prompt, size="1024x1024")
    images.append(response.consume())

print(f"Generated {len(images)} variations")
```

### Iterative Refinement

```python
import msgflux as mf

model = mf.Model.text_to_image("openai/dall-e-3")

# Start with base concept
prompt = "A robot"

# Iteratively add details
refinements = [
    "A humanoid robot",
    "A humanoid robot in a workshop",
    "A humanoid robot in a workshop, holding tools",
    "A humanoid robot in a workshop, holding tools, dramatic lighting"
]

for i, refined_prompt in enumerate(refinements):
    response = model(prompt=refined_prompt)
    print(f"Iteration {i+1}: {response.consume()}")
    print(f"Revised: {response.metadata.get('revised_prompt', 'N/A')}\n")
```

## Best Practices

### 1. Be Specific and Detailed

```python
# Good - Specific details
prompt = """A professional food photograph of a gourmet burger:
- Brioche bun with sesame seeds
- Perfectly grilled beef patty
- Fresh lettuce, tomato, and onion
- Melted cheddar cheese
- Wooden cutting board
- Natural daylight from window
- Shallow depth of field"""

# Less effective - Vague
prompt = "A burger"
```

### 2. Specify Technical Aspects

```python
# Include photography/art technical details
prompts = [
    # Photography
    "Portrait photo, 85mm lens, f/1.8, bokeh background, golden hour",

    # Digital art
    "Digital illustration, flat design, vibrant colors, vector style",

    # 3D render
    "3D render, physically based rendering, ray tracing, high detail"
]
```

### 3. Use Quality Parameters Wisely

```python
import msgflux as mf

model = mf.Model.text_to_image("openai/dall-e-3")

# For final/client work - HD quality
client_image = model(
    prompt="Professional product photo",
    quality="hd",
    size="1792x1024"
)

# For iterations/drafts - Standard quality
draft_image = model(
    prompt="Quick concept sketch",
    quality="standard",
    size="1024x1024"
)
```

### 4. Handle Errors Gracefully

```python
import msgflux as mf

model = mf.Model.text_to_image("openai/dall-e-3")

def safe_generate(prompt, max_retries=3):
    """Generate with error handling and retries."""
    for attempt in range(max_retries):
        try:
            response = model(prompt=prompt)
            return response.consume()
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
    return None

image_url = safe_generate("A beautiful landscape")
```

### 5. Respect Content Policies

```python
# Good - Follows content policies
safe_prompts = [
    "A family having dinner together",
    "Children playing in a park",
    "A doctor examining a patient"
]

# Avoid - May violate policies
# - Violence, gore
# - Explicit content
# - Illegal activities
# - Copyrighted characters
# - Public figures (use generic descriptions)
```

## Cost Optimization

### Choose Appropriate Model

```python
import msgflux as mf

# DALL-E 3: Higher quality, higher cost
# Use for: Final outputs, client work, high-detail needs
model_hd = mf.Model.text_to_image("openai/dall-e-3")

# DALL-E 2: Lower cost, good quality
# Use for: Iterations, drafts, bulk generation
model_standard = mf.Model.text_to_image("openai/dall-e-2")
```

### Batch Similar Requests

```python
import msgflux as mf
import msgflux.nn.functional as F

model = mf.Model.text_to_image("openai/dall-e-2")

# Generate multiple similar images in one batch
prompts = [f"Product photo of {item}" for item in [
    "coffee mug",
    "water bottle",
    "notebook",
    "pen"
]]

results = F.map_gather(
    model,
    args_list=[(p,) for p in prompts]
)
```

## Error Handling

```python
import msgflux as mf

model = mf.Model.text_to_image("openai/dall-e-3")

try:
    response = model(prompt="A landscape")
    image_url = response.consume()
except ImportError:
    print("Provider not installed")
except ValueError as e:
    print(f"Invalid parameters: {e}")
except Exception as e:
    print(f"Generation failed: {e}")
    # Common errors:
    # - Content policy violation
    # - Rate limits
    # - Network issues
```

## See Also

- [Model](model.md) - Model factory and registry
- [Image Editing](image_text_to_image.md) - Edit existing images with text
- [Chat Completion](chat_completion.md) - Generate image descriptions
- [Moderation](moderation.md) - Content moderation

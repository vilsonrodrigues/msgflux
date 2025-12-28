# Image Editing

The `image_text_to_image` model edits existing images using text descriptions. This enables modifications, inpainting, object removal, style changes, and creative transformations of existing visual content.

**All code examples use the recommended import pattern:**

```python
import msgflux as mf
```

## Overview

Image editing models take an existing image and modify it based on text prompts. They enable:

- **Image Modification**: Edit specific parts of an image
- **Inpainting**: Fill in or replace masked regions
- **Object Removal**: Remove unwanted elements
- **Style Transfer**: Change artistic style while preserving content
- **Creative Editing**: Add, modify, or transform elements

### Common Use Cases

- **Photo Editing**: Remove objects, change backgrounds
- **Product Photography**: Modify product colors, backgrounds, settings
- **Content Creation**: Transform existing images for marketing
- **Image Restoration**: Fill in missing or damaged areas
- **Creative Variations**: Generate alternative versions of images

## Quick Start

### Basic Usage

```python
import msgflux as mf

# Create image editor
model = mf.Model.image_text_to_image("openai/dall-e-2")

# Edit image
response = model(
    prompt="Add a sunset sky",
    image="path/to/image.png"
)

# Get edited image URL
edited_url = response.consume()
print(edited_url)  # https://...
```

### With Mask (Inpainting)

```python
import msgflux as mf

model = mf.Model.image_text_to_image("openai/dall-e-2")

# Edit only masked area
response = model(
    prompt="A wooden table",
    image="room.png",
    mask="table_mask.png"  # Transparent areas will be edited
)

edited_url = response.consume()
```

## Supported Providers

### OpenAI (DALL-E 2)

Currently, image editing is primarily supported by DALL-E 2:

```python
import msgflux as mf

# DALL-E 2 for image editing
model = mf.Model.image_text_to_image("openai/dall-e-2")
```

Note: DALL-E 3 does not support image editing. Use DALL-E 2 for this functionality.

### Replicate

```python
import msgflux as mf

# Various Replicate models support image editing
model = mf.Model.image_text_to_image("replicate/stability-ai/stable-diffusion-inpainting")
```

## How Image Editing Works

### Without Mask

When no mask is provided, the model edits the entire image:

```python
import msgflux as mf

model = mf.Model.image_text_to_image("openai/dall-e-2")

# Edit entire image
response = model(
    prompt="Make it look like a watercolor painting",
    image="photo.jpg"
)
```

### With Mask (Inpainting)

Masks define which areas to edit:

```python
import msgflux as mf

model = mf.Model.image_text_to_image("openai/dall-e-2")

# Only edit masked areas
response = model(
    prompt="A blue sky with clouds",
    image="landscape.png",
    mask="sky_mask.png"  # Transparent PNG showing sky area
)
```

**Mask Requirements:**
- Must be a PNG image with transparency (alpha channel)
- Fully transparent areas (alpha = 0) indicate where to edit
- Same dimensions as the input image (1024x1024 for DALL-E 2)
- Opaque areas remain unchanged

## Image Input Formats

The `image` parameter accepts multiple formats:

### File Path

```python
import msgflux as mf

model = mf.Model.image_text_to_image("openai/dall-e-2")

response = model(
    prompt="Change background to beach",
    image="/path/to/photo.png"
)
```

### URL

```python
import msgflux as mf

model = mf.Model.image_text_to_image("openai/dall-e-2")

response = model(
    prompt="Add winter atmosphere",
    image="https://example.com/photo.jpg"
)
```

### Base64 String

```python
import msgflux as mf
import base64

# Read and encode image
with open("photo.png", "rb") as f:
    img_data = base64.b64encode(f.read()).decode()

model = mf.Model.image_text_to_image("openai/dall-e-2")

response = model(
    prompt="Make it artistic",
    image=f"data:image/png;base64,{img_data}"
)
```

## Response Formats

### URL Response (Default)

```python
import msgflux as mf

model = mf.Model.image_text_to_image("openai/dall-e-2")

response = model(
    prompt="Add flowers in foreground",
    image="garden.png",
    response_format="url"  # Default
)

# Get URL
url = response.consume()
print(url)  # https://...

# Download
import requests
img_data = requests.get(url).content
with open("edited.png", "wb") as f:
    f.write(img_data)
```

### Base64 Response

```python
import msgflux as mf
import base64

model = mf.Model.image_text_to_image("openai/dall-e-2")

response = model(
    prompt="Change to evening lighting",
    image="scene.png",
    response_format="base64"
)

# Get base64 data
b64_data = response.consume()

# Decode and save
img_data = base64.b64decode(b64_data)
with open("edited.png", "wb") as f:
    f.write(img_data)
```

## Multiple Variations

Generate multiple edited versions:

```python
import msgflux as mf

model = mf.Model.image_text_to_image("openai/dall-e-2")

# Generate 4 variations
response = model(
    prompt="Add dramatic storm clouds",
    image="landscape.jpg",
    n=4  # Number of variations
)

# Get all URLs
edited_images = response.consume()
print(f"Generated {len(edited_images)} variations")

for i, url in enumerate(edited_images):
    print(f"Variation {i+1}: {url}")
```

## Image Size Requirements

### DALL-E 2 Requirements

```python
import msgflux as mf

model = mf.Model.image_text_to_image("openai/dall-e-2")

# Image must be:
# - Square (1024x1024 recommended)
# - PNG format
# - Less than 4MB
# - If using mask, same size as image

response = model(
    prompt="Edit the background",
    image="1024x1024_image.png"
)
```

## Creating Masks

### Using Image Editing Software

Create masks in tools like Photoshop, GIMP, or programmatically:

```python
from PIL import Image
import numpy as np

# Load original image
img = Image.open("photo.png").convert("RGBA")
width, height = img.size

# Create mask (transparent where you want to edit)
mask = Image.new("RGBA", (width, height), (255, 255, 255, 255))
pixels = mask.load()

# Make a rectangular region transparent (will be edited)
for x in range(200, 800):
    for y in range(100, 500):
        pixels[x, y] = (0, 0, 0, 0)  # Fully transparent

mask.save("edit_mask.png")
```

### Programmatic Mask Creation

```python
from PIL import Image, ImageDraw

# Create 1024x1024 mask
mask = Image.new("RGBA", (1024, 1024), (255, 255, 255, 255))
draw = ImageDraw.Draw(mask)

# Draw transparent circle (area to edit)
draw.ellipse(
    [(200, 200), (800, 800)],
    fill=(0, 0, 0, 0)  # Transparent
)

mask.save("circle_mask.png")
```

## Common Editing Tasks

### Background Removal/Replacement

```python
import msgflux as mf

model = mf.Model.image_text_to_image("openai/dall-e-2")

# Replace background
response = model(
    prompt="Professional studio background with soft lighting",
    image="portrait.png",
    mask="background_mask.png"  # Background area transparent
)
```

### Object Removal

```python
import msgflux as mf

model = mf.Model.image_text_to_image("openai/dall-e-2")

# Remove object by filling with surrounding context
response = model(
    prompt="Natural grass lawn",
    image="yard.png",
    mask="object_mask.png"  # Object area transparent
)
```

### Style Transfer

```python
import msgflux as mf

model = mf.Model.image_text_to_image("openai/dall-e-2")

# Change entire image style
response = model(
    prompt="Oil painting in impressionist style, vibrant colors",
    image="photo.jpg"
    # No mask = edit entire image
)
```

### Adding Elements

```python
import msgflux as mf

model = mf.Model.image_text_to_image("openai/dall-e-2")

# Add new elements to specific area
response = model(
    prompt="A red sports car parked",
    image="driveway.png",
    mask="parking_spot_mask.png"
)
```

### Color/Lighting Adjustments

```python
import msgflux as mf

model = mf.Model.image_text_to_image("openai/dall-e-2")

# Change lighting/atmosphere
response = model(
    prompt="Warm sunset lighting, golden hour atmosphere",
    image="scene.jpg"
)
```

## Async Support

Edit images asynchronously:

```python
import msgflux as mf
import asyncio

model = mf.Model.image_text_to_image("openai/dall-e-2")

async def edit_image(image_path, prompt):
    response = await model.acall(
        prompt=prompt,
        image=image_path
    )
    return response.consume()

async def main():
    # Edit multiple images concurrently
    edits = [
        ("photo1.png", "Add sunset sky"),
        ("photo2.png", "Change to winter scene"),
        ("photo3.png", "Make it vintage style")
    ]

    tasks = [edit_image(img, prompt) for img, prompt in edits]
    results = await asyncio.gather(*tasks)

    for (img, prompt), url in zip(edits, results):
        print(f"{img} ({prompt}): {url}")

asyncio.run(main())
```

## Batch Processing

Edit multiple images:

```python
import msgflux as mf
import msgflux.nn.functional as F

model = mf.Model.image_text_to_image("openai/dall-e-2")

images = ["photo1.png", "photo2.png", "photo3.png"]
prompt = "Professional studio background"

# Process in parallel
results = F.map_gather(
    model,
    args_list=[
        (prompt, img) for img in images
    ]
)

# Get all edited URLs
edited_urls = [r.consume() for r in results]

for original, edited in zip(images, edited_urls):
    print(f"{original} -> {edited}")
```

## Best Practices

### 1. Prepare Images Correctly

```python
from PIL import Image

def prepare_image_for_editing(image_path, output_path):
    """Prepare image for DALL-E 2 editing."""
    img = Image.open(image_path)

    # Convert to RGBA
    img = img.convert("RGBA")

    # Resize to 1024x1024
    img = img.resize((1024, 1024), Image.Resampling.LANCZOS)

    # Save as PNG
    img.save(output_path, "PNG")

    return output_path

# Use prepared image
prepared = prepare_image_for_editing("original.jpg", "prepared.png")

import msgflux as mf
model = mf.Model.image_text_to_image("openai/dall-e-2")
response = model(prompt="Edit this", image=prepared)
```

### 2. Create Effective Masks

```python
# Good - Clear boundaries
def create_clean_mask(image_path, region):
    """Create a clean mask with feathered edges."""
    from PIL import Image, ImageDraw, ImageFilter

    img = Image.open(image_path)
    mask = Image.new("L", img.size, 255)  # Start opaque
    draw = ImageDraw.Draw(mask)

    # Make region transparent
    draw.rectangle(region, fill=0)

    # Feather edges for smooth blend
    mask = mask.filter(ImageFilter.GaussianBlur(5))

    # Convert to RGBA
    rgba_mask = Image.new("RGBA", img.size, (255, 255, 255, 255))
    rgba_mask.putalpha(mask)

    return rgba_mask

mask = create_clean_mask("photo.png", (100, 100, 900, 900))
mask.save("smooth_mask.png")
```

### 3. Write Contextual Prompts

```python
import msgflux as mf

model = mf.Model.image_text_to_image("openai/dall-e-2")

# Good - Describes what should be in the edited area
response = model(
    prompt="Modern glass coffee table with magazines and a vase of flowers",
    image="living_room.png",
    mask="table_area_mask.png"
)

# Less effective - Too vague
response = model(
    prompt="Nice furniture",
    image="living_room.png",
    mask="table_area_mask.png"
)
```

### 4. Generate Multiple Variations

```python
import msgflux as mf

model = mf.Model.image_text_to_image("openai/dall-e-2")

# Generate multiple options to choose from
response = model(
    prompt="Dramatic storm clouds at sunset",
    image="landscape.png",
    mask="sky_mask.png",
    n=4  # Generate 4 variations
)

variations = response.consume()

# Review and pick the best one
for i, url in enumerate(variations):
    print(f"Option {i+1}: {url}")
```

### 5. Preserve Image Context

```python
import msgflux as mf

model = mf.Model.image_text_to_image("openai/dall-e-2")

# Good - Maintains context and consistency
response = model(
    prompt="Matching wooden chair in the same modern minimalist style",
    image="room_with_table.png",
    mask="empty_corner_mask.png"
)

# Less effective - Ignores existing style
response = model(
    prompt="A chair",
    image="room_with_table.png",
    mask="empty_corner_mask.png"
)
```

## Common Patterns

### Before/After Comparison

```python
import msgflux as mf
from io import BytesIO

def edit_and_compare(original_path, prompt, mask_path=None):
    """Edit image and save before/after."""
    from PIL import Image
    import requests

    model = mf.Model.image_text_to_image("openai/dall-e-2")

    # Edit
    response = model(
        prompt=prompt,
        image=original_path,
        mask=mask_path
    )

    # Download edited
    edited_url = response.consume()
    edited_data = requests.get(edited_url).content

    # Create side-by-side comparison
    original = Image.open(original_path)
    edited = Image.open(BytesIO(edited_data))

    # Combine
    comparison = Image.new("RGB", (2048, 1024))
    comparison.paste(original, (0, 0))
    comparison.paste(edited, (1024, 0))

    comparison.save("comparison.png")
    return edited_url

edit_and_compare("photo.png", "Change to autumn colors")
```

### Iterative Editing

```python
import msgflux as mf
import requests

model = mf.Model.image_text_to_image("openai/dall-e-2")

def download_image(url, path):
    """Download image from URL."""
    data = requests.get(url).content
    with open(path, "wb") as f:
        f.write(data)
    return path

# Step 1: Change background
response1 = model(
    prompt="Beach sunset background",
    image="original.png",
    mask="background_mask.png"
)
step1_path = download_image(response1.consume(), "step1.png")

# Step 2: Adjust lighting on result
response2 = model(
    prompt="Warm golden hour lighting",
    image=step1_path
)
final_path = download_image(response2.consume(), "final.png")
```

### Batch Product Editing

```python
import msgflux as mf
import msgflux.nn.functional as F

model = mf.Model.image_text_to_image("openai/dall-e-2")

products = [
    ("product1.png", "Clean white background"),
    ("product2.png", "Clean white background"),
    ("product3.png", "Clean white background")
]

# Edit all product backgrounds in parallel
results = F.map_gather(
    model,
    args_list=[
        (prompt, img, "background_mask.png")
        for img, prompt in products
    ]
)

# Save results
for i, result in enumerate(results):
    url = result.consume()
    print(f"Product {i+1} edited: {url}")
```

## Error Handling

```python
import msgflux as mf

model = mf.Model.image_text_to_image("openai/dall-e-2")

try:
    response = model(
        prompt="Edit this image",
        image="photo.png",
        mask="mask.png"
    )
    url = response.consume()
except ImportError:
    print("Provider not installed")
except ValueError as e:
    print(f"Invalid parameters: {e}")
    # Common issues:
    # - Image not square
    # - Image too large (>4MB)
    # - Mask doesn't match image size
    # - Invalid image format
except Exception as e:
    print(f"Edit failed: {e}")
    # Common errors:
    # - Content policy violation
    # - Rate limits
    # - Network issues
```

## Limitations

### DALL-E 2 Limitations

- **Size**: Images must be square (1024x1024)
- **Format**: PNG required for masks
- **File Size**: <4MB for images
- **Model**: Only DALL-E 2 supports editing (DALL-E 3 does not)

### Quality Considerations

```python
# For best results:
# - Use high-quality source images
# - Create clean masks with smooth edges
# - Provide detailed, contextual prompts
# - Generate multiple variations
# - Consider the existing image style and lighting
```

## Cost Optimization

### Efficient Editing Workflow

```python
import msgflux as mf

# DALL-E 2 is cost-effective for editing
model = mf.Model.image_text_to_image("openai/dall-e-2")

# Generate fewer variations initially
response = model(
    prompt="New background",
    image="photo.png",
    n=2  # Start with 2 variations
)

# If not satisfied, generate more
if not_satisfied:
    response = model(
        prompt="New background, more dramatic",
        image="photo.png",
        n=2
    )
```

## See Also

- [Text to Image](text_to_image.md) - Generate images from scratch
- [Model](model.md) - Model factory and registry
- [Chat Completion](chat_completion.md) - Generate descriptive prompts 
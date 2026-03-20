# Image Editing

The `image_text_to_image` model edits existing images using text descriptions. This enables modifications, inpainting, object removal, style changes, and creative transformations of existing visual content.

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

???+ example

    ```python
    import msgflux as mf

    # Create image editor
    model = mf.Model.image_text_to_image("openai/gpt-image-1")

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

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

    # Edit only masked area
    response = model(
        prompt="A wooden table",
        image="room.png",
        mask="table_mask.png"  # Transparent areas will be edited
    )

    edited_url = response.consume()
    ```

## Supported Providers

### OpenAI

???+ example

    ```python
    import msgflux as mf

    # Latest model — precise editing with logo and face preservation
    model = mf.Model.image_text_to_image("openai/gpt-image-1.5")

    # Stable release
    model = mf.Model.image_text_to_image("openai/gpt-image-1")
    ```

### Replicate

???+ example

    ```python
    import msgflux as mf

    # Various Replicate models support image editing
    model = mf.Model.image_text_to_image("replicate/stability-ai/stable-diffusion-inpainting")
    ```

## How Image Editing Works

### Without Mask

When no mask is provided, the model edits the entire image:

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

    # Edit entire image
    response = model(
        prompt="Make it look like a watercolor painting",
        image="photo.jpg"
    )
    ```

### With Mask (Inpainting)

Masks define which areas to edit:

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

    # Only edit masked areas
    response = model(
        prompt="A blue sky with clouds",
        image="landscape.png",
        mask="sky_mask.png"  # Transparent PNG showing sky area
    )
    ```

The mask is a hint to the model alongside the prompt — masking with `gpt-image-1` is entirely prompt-based. The image must be a PNG and the mask (if provided) the same dimensions as the input.

## Image Input Formats

The `image` parameter accepts multiple formats:

### File Path

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

    response = model(
        prompt="Change background to beach",
        image="/path/to/photo.png"
    )
    ```

### URL

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

    response = model(
        prompt="Add winter atmosphere",
        image="https://example.com/photo.jpg"
    )
    ```

### Base64 String

???+ example

    ```python
    import msgflux as mf
    import base64

    # Read and encode image
    with open("photo.png", "rb") as f:
        img_data = base64.b64encode(f.read()).decode()

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

    response = model(
        prompt="Make it artistic",
        image=f"data:image/png;base64,{img_data}"
    )
    ```

## Response Formats

### URL Response (Default)

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

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

???+ example

    ```python
    import msgflux as mf
    import base64

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

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

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

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

### gpt-image-1 Requirements

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

    # Image requirements:
    # - PNG, WEBP, or JPEG format
    # - Less than 25MB
    # - If using mask, same dimensions as the image
    # Supported output sizes: 1024x1024, 1536x1024, 1024x1536, auto

    response = model(
        prompt="Edit the background",
        image="photo.png",
        size="1024x1024"
    )
    ```

## Creating Masks

With `gpt-image-1`, masking is **prompt-based**: the mask tells the model *where* to focus, and the prompt tells it *what* to do.

### Programmatic Mask Creation

???+ example

    ```python
    from PIL import Image, ImageDraw

    # Create a mask the same size as the input image
    # White = keep, Black = edit region (for gpt-image-1, this is a hint)
    img = Image.open("photo.png")
    mask = Image.new("RGBA", img.size, (255, 255, 255, 255))
    draw = ImageDraw.Draw(mask)

    # Mark the region to edit as transparent
    draw.rectangle([200, 100, 800, 500], fill=(0, 0, 0, 0))

    mask.save("edit_mask.png")
    ```

### Prompt-Only Editing (no mask)

With `gpt-image-1` you can edit without a mask — the model infers the region from the prompt:

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

    # The model understands "the background" from the prompt alone
    response = model(
        prompt="Replace the background with a sunset beach",
        image="portrait.png"
    )
    ```

## Common Editing Tasks

### Background Removal/Replacement

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

    # Replace background
    response = model(
        prompt="Professional studio background with soft lighting",
        image="portrait.png",
        mask="background_mask.png"  # Background area transparent
    )
    ```

### Object Removal

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

    # Remove object by filling with surrounding context
    response = model(
        prompt="Natural grass lawn",
        image="yard.png",
        mask="object_mask.png"  # Object area transparent
    )
    ```

### Style Transfer

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

    # Change entire image style
    response = model(
        prompt="Oil painting in impressionist style, vibrant colors",
        image="photo.jpg"
        # No mask = edit entire image
    )
    ```

### Adding Elements

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

    # Add new elements to specific area
    response = model(
        prompt="A red sports car parked",
        image="driveway.png",
        mask="parking_spot_mask.png"
    )
    ```

### Color/Lighting Adjustments

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

    # Change lighting/atmosphere
    response = model(
        prompt="Warm sunset lighting, golden hour atmosphere",
        image="scene.jpg"
    )
    ```

## Async Support

Edit images asynchronously:

???+ example

    ```python
    import msgflux as mf
    import asyncio

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

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

???+ example

    ```python
    import msgflux as mf
    import msgflux.nn.functional as F

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

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

???+ example

    ```python
    from PIL import Image

    def prepare_image_for_editing(image_path, output_path):
        """Prepare image for gpt-image-1 editing."""
        img = Image.open(image_path)

        # Convert to RGBA for PNG output
        img = img.convert("RGBA")

        # Resize to a supported size (1024x1024 is a safe default)
        img = img.resize((1024, 1024), Image.Resampling.LANCZOS)

        # Save as PNG
        img.save(output_path, "PNG")

        return output_path

    # Use prepared image
    prepared = prepare_image_for_editing("original.jpg", "prepared.png")

    import msgflux as mf
    model = mf.Model.image_text_to_image("openai/gpt-image-1")
    response = model(prompt="Edit this", image=prepared)
    ```

### 2. Create Effective Masks

???+ example

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

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

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

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

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

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

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

???+ example

    ```python
    import msgflux as mf
    from io import BytesIO

    def edit_and_compare(original_path, prompt, mask_path=None):
        """Edit image and save before/after."""
        from PIL import Image
        import requests

        model = mf.Model.image_text_to_image("openai/gpt-image-1")

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

???+ example

    ```python
    import msgflux as mf
    import requests

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

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

???+ example

    ```python
    import msgflux as mf
    import msgflux.nn.functional as F

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

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

???+ example

    ```python
    import msgflux as mf

    model = mf.Model.image_text_to_image("openai/gpt-image-1")

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

### gpt-image-1 Limitations

- **Format**: PNG, WEBP, or JPEG for input; PNG for masks
- **File Size**: Up to 25MB per image
- **Output sizes**: 1024x1024, 1536x1024, 1024x1536, or `auto`
- **Masking**: Prompt-based — complex pixel-perfect edits may need multiple iterations

### Quality Considerations

???+ example

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

???+ example

    ```python
    import msgflux as mf

    # gpt-image-1 is the recommended model for all editing tasks
    model = mf.Model.image_text_to_image("openai/gpt-image-1")

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

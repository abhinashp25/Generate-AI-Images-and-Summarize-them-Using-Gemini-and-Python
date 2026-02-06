import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from vertexai.generative_models import GenerativeModel, Part


def generate_bouquet_image(prompt: str) -> str:
    # Initialize Vertex AI
    vertexai.init()

    # Load Imagen model for image generation
    model = ImageGenerationModel.from_pretrained(
        "imagen-4.0-generate-001"
    )

    # Generate bouquet image
    images = model.generate_images(
        prompt=prompt,
        number_of_images=1
    )

    # Save the generated image locally
    image_path = "bouquet.jpeg"
    images[0].save(image_path)

    print(f"Image generated and saved as {image_path}")
    return image_path


def analyze_bouquet_image(image_path: str):
    # Load Gemini multimodal model
    model = GenerativeModel("gemini-2.5-flash")

    # Read image as binary data
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    # Convert image to Part object
    image_part = Part.from_data(
        data=image_bytes,
        mime_type="image/jpeg"
    )

    # Prompt for image analysis
    prompt = (
        "Analyze this bouquet image and generate a short birthday wish "
        "based on the flowers you see."
    )

    # Generate response (streaming disabled as required)
    response = model.generate_content(
        [prompt, image_part],
        stream=False
    )

    print("Birthday wish:")
    print(response.text)


if __name__ == "__main__":
    prompt = "Create an image containing a bouquet of 2 sunflowers and 3 roses"
    image_path = generate_bouquet_image(prompt)
    analyze_bouquet_image(image_path)

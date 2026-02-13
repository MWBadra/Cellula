from PIL import Image
import httpx
from io import BytesIO
from transformers import AutoProcessor, BlipForConditionalGeneration

print("Loading BLIP Model...")
processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
print("Model Loaded.")

def generate_caption(url):
    try:
        with httpx.stream("GET", url) as response:
            image = Image.open(BytesIO(response.read()))
        text = "A picture of "

        inputs = processor(images=image, text=text, return_tensors="pt")

        # 5. GENERATE (The fix!)
        # We use .generate() to create new tokens
        outputs = model.generate(**inputs, max_new_tokens=50)

        # 6. Decode back to English
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        
        return caption

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    print(generate_caption(url))
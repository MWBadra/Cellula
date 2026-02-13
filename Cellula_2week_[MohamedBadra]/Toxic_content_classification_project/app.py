import os
from dotenv import load_dotenv 
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from peft import PeftModel
from fastapi.middleware.cors import CORSMiddleware
import imagecaption 
import motor.motor_asyncio
from datetime import datetime
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles # Optional, but good practice
# --- 1. SECURITY SETUP ---
# Load secrets from the .env file (only works on your computer)
load_dotenv()

# Get the MongoDB URL from the environment
MONGO_URL = os.getenv("MONGO_URL")

# Fallback: If no secret is found, try localhost (useful for testing without internet)
if not MONGO_URL:
    print("⚠️ WARNING: MONGO_URL not found. Using localhost.")
    MONGO_URL = "mongodb://localhost:27017"

# --- 2. DATABASE CONNECTION ---
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
db = client["toxic_content_db"]
collection = db["history"]

# --- 3. APP SETUP ---
app = FastAPI(title="Toxic Content Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# YOUR CATEGORIES (Must match training!)
CATEGORIES = [
    'Safe', 
    'Violent Crimes', 
    'Elections', 
    'Sex-Related Crimes', 
    'unsafe', 
    'Non-Violent Crimes', 
    'Child Sexual Exploitation', 
    'Unknown S-Type', 
    'Suicide & Self-Harm'
]

# --- 4. LOAD MODELS ---
print("Loading Models...")
base_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(CATEGORIES))

try:
    model = PeftModel.from_pretrained(base_model, "final_toxic_model")
    model.eval()
except:
    print("Warning: 'final_toxic_model' not found. Using base model.")
    model = base_model

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
print("Models Loaded!")

# --- 5. HELPER FUNCTION ---
def get_prediction(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    predicted_class_id = logits.argmax().item()
    confidence = torch.softmax(logits, dim=1)[0][predicted_class_id].item()
    predicted_label = CATEGORIES[predicted_class_id]
    return predicted_label, confidence

# --- 6. ENDPOINTS ---
class TextRequest(BaseModel):
    text: str

class UrlRequest(BaseModel):
    image_url: str

@app.post("/predict")
async def predict_text(request: TextRequest):
    try:
        label, conf = get_prediction(request.text)
        is_toxic = label != "Safe"

        # Save to Cloud DB
        document = {
            "type": "text",
            "input": request.text,
            "prediction": label,
            "confidence": conf,
            "is_toxic": is_toxic,
            "timestamp": datetime.now()
        }
        await collection.insert_one(document)

        return {"prediction": label, "confidence": f"{conf:.2f}", "is_toxic": is_toxic}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_url")
async def analyze_url(request: UrlRequest):
    try:
        caption = imagecaption.generate_caption(request.image_url)
        if caption.startswith("Error"):
            raise HTTPException(status_code=400, detail=caption)

        label, conf = get_prediction(caption)
        is_toxic = label != "Safe"

        # Save to Cloud DB
        document = {
            "type": "image",
            "image_url": request.image_url,
            "generated_caption": caption,
            "prediction": label,
            "confidence": conf,
            "is_toxic": is_toxic,
            "timestamp": datetime.now()
        }
        await collection.insert_one(document)

        return {
            "caption": caption,
            "prediction": label,
            "confidence": f"{conf:.2f}",
            "is_toxic": is_toxic
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
async def get_history():
    try:
        cursor = collection.find().sort("timestamp", -1).limit(50)
        history_list = await cursor.to_list(length=50)

        formatted_history = []
        for doc in history_list:
            doc["_id"] = str(doc["_id"])
            
            if "timestamp" in doc:
                doc["timestamp"] = doc["timestamp"].strftime("%Y-%m-%d %H:%M")
            
            formatted_history.append(doc)
            
        return formatted_history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_index():
    return FileResponse('index.html')
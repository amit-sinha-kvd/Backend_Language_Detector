import os
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(title="Language Detector API")

# Has CORS enabled for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request body schema
class InputData(BaseModel):
    text: str

# Global model variable
model = None

def load_or_train_model():
    global model
    # Loads model.pkl using joblib on startup
    if os.path.exists('model.pkl'):
        print("Loading existing model...")
        model = joblib.load('model.pkl')
    else:
        # If model.pkl does not exist, trains and saves it automatically
        print("Model not found. Initializing training...")
        import model_train
        model = joblib.load('model.pkl')

@app.on_event("startup")
async def startup_event():
    load_or_train_model()

# POST endpoint at /predict
@app.post("/predict")
async def predict(data: InputData):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not data.text.strip():
        return {"prediction": "Unknown", "confidence": 0.0}

    # Accepts input as JSON body and returns prediction
    # model.predict expects a list
    prediction = model.predict([data.text])[0]
    
    # Simple confidence score (probability)
    probs = model.predict_proba([data.text])[0]
    confidence = float(max(probs))

    return {
        "prediction": prediction,
        "confidence": confidence
    }

if __name__ == "__main__":
    import uvicorn
    # The start command logic
    port = int(os.getenv("PORT", 3000))
    uvicorn.run(app, host="0.0.0.0", port=port)

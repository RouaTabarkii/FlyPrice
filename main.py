from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from backend.main import app as prediction_app, load_model
from backend.auth import auth_app
from backend.recommendation import recommendation_app
from backend.chatbot import chatbot_app

app = FastAPI(
    title="Flight Price Prediction System",
    description="Complete flight price prediction and travel assistance system",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/predict", prediction_app)
app.mount("/auth", auth_app)
app.mount("/recommend", recommendation_app)
app.mount("/chat", chatbot_app)

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Flight Price Prediction System",
        "version": "1.0.0",
        "services": {
            "prediction": "/predict",
            "authentication": "/auth",
            "recommendation": "/recommend",
            "chatbot": "/chat"
        },
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }

@app.get("/health")
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "services": {
            "prediction": "active",
            "authentication": "active",
            "recommendation": "active",
            "chatbot": "active"
        },
        "timestamp": "2024-01-01T00:00:00Z"
    }

if __name__ == "__main__":
    print("Starting Flight Price Prediction System...")
    print("Services available at:")
    print("  - Prediction API: http://localhost:8000/predict")
    print("  - Auth API: http://localhost:8000/auth")
    print("  - Recommendation API: http://localhost:8000/recommend")
    print("  - Chatbot API: http://localhost:8000/chat")
    print("  - API Docs: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

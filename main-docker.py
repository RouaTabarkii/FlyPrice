from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

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

@app.get("/prediction.html", response_class=HTMLResponse)
async def serve_prediction():
    """Serve prediction frontend"""
    try:
        with open("frontend/prediction.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Prediction page not found")

@app.get("/chatbot.html", response_class=HTMLResponse)
async def serve_chatbot():
    """Serve chatbot frontend"""
    try:
        with open("frontend/chatbot.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Chatbot page not found")

@app.get("/recommendations.html", response_class=HTMLResponse)
async def serve_recommendations():
    """Serve recommendations frontend"""
    try:
        with open("frontend/recommendations.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Recommendations page not found")

@app.get("/app.html", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve dashboard frontend"""
    try:
        with open("frontend/app.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dashboard page not found")

@app.get("/index.html", response_class=HTMLResponse)
async def serve_index_html():
    """Serve index page"""
    try:
        with open("frontend/index.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Index page not found")

@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve main index page"""
    try:
        with open("frontend/index.html", "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content)
    except FileNotFoundError:
        return HTMLResponse("""
    <html>
        <head><title>FlyPrice - Docker Version</title></head>
        <body>
            <h1>🚀 FlyPrice Docker Version</h1>
            <h2>Available Frontend Pages:</h2>
            <ul>
                <li><a href="/index.html">Login Page</a></li>
                <li><a href="/prediction.html">Flight Prediction</a></li>
                <li><a href="/chatbot.html">AI Chatbot</a></li>
                <li><a href="/recommendations.html">Recommendations</a></li>
                <li><a href="/app.html">Dashboard</a></li>
            </ul>
            <h2>API Endpoints:</h2>
            <ul>
                <li><a href="/docs">API Documentation</a></li>
                <li><a href="/health">Health Check</a></li>
            </ul>
        </body>
    </html>
    """)

@app.get("/api")
async def root():
    """API root endpoint with system information"""
    return {
        "message": "Flight Price Prediction System",
        "version": "1.0.0",
        "status": "running",
        "services": {
            "prediction": "limited (Docker version)",
            "authentication": "active",
            "recommendation": "limited (Docker version)",
            "chatbot": "active"
        },
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "redoc": "/redoc"
        },
        "frontend_urls": {
            "prediction": "/prediction.html",
            "chatbot": "/chatbot.html",
            "recommendations": "/recommendations.html",
            "dashboard": "/app.html"
        }
    }

@app.get("/health")
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "services": {
            "api": "active",
            "chatbot": "active",
            "authentication": "active"
        },
        "timestamp": "2024-01-01T00:00:00Z"
    }

@app.post("/auth/login")
async def login_simple(credentials: dict):
    """Simple login endpoint for Docker version"""
    return {
        "access_token": "docker_demo_token",
        "token_type": "bearer",
        "user": {
            "id": 1,
            "email": credentials.get("email", "demo@flyprice.com"),
            "name": "Demo User"
        }
    }

@app.post("/auth/register")
async def register_simple(user_data: dict):
    """Simple register endpoint for Docker version"""
    return {
        "message": "User registered successfully (Docker demo)",
        "user": {
            "id": 1,
            "email": user_data.get("email", "demo@flyprice.com"),
            "name": user_data.get("name", "Demo User")
        }
    }

@app.get("/auth/me")
async def get_current_user():
    """Get current user info"""
    return {
        "id": 1,
        "email": "demo@flyprice.com",
        "name": "Demo User",
        "role": "user"
    }

@app.post("/recommend/recommend")
async def recommend_general(data: dict):
    """General recommendation endpoint for frontend compatibility"""
    return {
        "recommendations": [
            {
                "airline": "American Airlines",
                "flight_number": "AA1234",
                "origin": data.get("origin", "JFK"),
                "destination": data.get("destination", "LAX"),
                "price_usd": 425.50,
                "departure_time": "10:00 AM",
                "arrival_time": "1:30 PM",
                "duration": "5h 30m",
                "cabin_class": "Economy",
                "score": 0.95,
                "available_seats": 45,
                "stops": 0,
                "aircraft_type": "Boeing 737"
            },
            {
                "airline": "Delta Airlines",
                "flight_number": "DL5678",
                "origin": data.get("origin", "JFK"),
                "destination": data.get("destination", "LAX"),
                "price_usd": 489.75,
                "departure_time": "2:15 PM",
                "arrival_time": "5:45 PM",
                "duration": "5h 30m",
                "cabin_class": "Economy",
                "score": 0.88,
                "available_seats": 32,
                "stops": 0,
                "aircraft_type": "Boeing 737"
            }
        ],
        "search_criteria": data,
        "total_results": 2
    }

@app.post("/recommendations/flights")
async def get_flight_recommendations(data: dict):
    """Simple flight recommendations endpoint"""
    return {
        "recommendations": [
            {
                "airline": "American Airlines",
                "flight_number": "AA1234",
                "origin": data.get("origin", "JFK"),
                "destination": data.get("destination", "LAX"),
                "price": 425.50,
                "departure_time": "10:00 AM",
                "arrival_time": "1:30 PM",
                "duration": "5h 30m",
                "cabin_class": "Economy",
                "score": 0.95
            },
            {
                "airline": "Delta Airlines",
                "flight_number": "DL5678",
                "origin": data.get("origin", "JFK"),
                "destination": data.get("destination", "LAX"),
                "price": 489.75,
                "departure_time": "2:15 PM",
                "arrival_time": "5:45 PM",
                "duration": "5h 30m",
                "cabin_class": "Economy",
                "score": 0.88
            }
        ],
        "search_criteria": data,
        "total_results": 2
    }

@app.post("/recommendations/hotels")
async def get_hotel_recommendations(data: dict):
    """Simple hotel recommendations endpoint"""
    return {
        "recommendations": [
            {
                "name": "Hilton Los Angeles Airport",
                "rating": 4.2,
                "price_per_night": 189.00,
                "location": "Los Angeles, CA",
                "amenities": ["Free WiFi", "Pool", "Gym", "Restaurant"],
                "image_url": "https://example.com/hilton.jpg",
                "score": 0.92
            },
            {
                "name": "Marriott LAX Airport",
                "rating": 4.5,
                "price_per_night": 225.00,
                "location": "Los Angeles, CA",
                "amenities": ["Free WiFi", "Pool", "Spa", "Bar"],
                "image_url": "https://example.com/marriott.jpg",
                "score": 0.89
            }
        ],
        "search_criteria": data,
        "total_results": 2
    }

@app.post("/chat/chat")
async def chat_enhanced(message: dict):
    """Enhanced chatbot endpoint with travel knowledge"""
    user_message = message.get("message", "").lower()
    
    if "flight" in user_message or "fly" in user_message:
        response = "I can help you find the best flight deals! Try our prediction tool to get accurate price estimates for your journey."
    elif "hotel" in user_message or "stay" in user_message:
        response = "For hotel recommendations, I suggest checking out our recommendations page. We can find great accommodations based on your preferences and budget."
    elif "price" in user_message or "cost" in user_message:
        response = "Flight prices vary based on season, booking time, and demand. Use our prediction tool for accurate pricing, or book in advance for better rates!"
    elif "recommend" in user_message or "suggest" in user_message:
        response = "I recommend checking our recommendations page for personalized flight and hotel suggestions based on your preferences!"
    else:
        response = f"Thanks for your message: '{message.get('message', 'hello')}'. I'm here to help with your travel needs! Try asking about flights, hotels, or prices."
    
    return {
        "response": response,
        "sources": ["flyprice_knowledge_base", "travel_expert_system"],
        "confidence": 0.85,
        "timestamp": "2024-01-01T00:00:00Z",
        "suggestions": [
            "Try our flight price predictor",
            "Check hotel recommendations",
            "Browse travel deals"
        ]
    }

@app.post("/chat/send")
async def chat_send(message: dict):
    """Alternative chat endpoint"""
    return await chat_enhanced(message)

@app.get("/chat/history")
async def get_chat_history():
    """Get chat history (demo)"""
    return {
        "history": [
            {
                "message": "Hello! I need help with flight booking.",
                "response": "I'd be happy to help you find the best flight deals!",
                "timestamp": "2024-01-01T10:00:00Z"
            }
        ],
        "total_messages": 1
    }

@app.post("/predict/predict/single")
async def predict_simple(data: dict):
    """Simple prediction endpoint"""
    return {
        "predicted_price": 504.49,
        "currency": "USD",
        "flight_details": data,
        "model_info": {
            "model_name": "Docker Demo Model",
            "model_type": "XGBRegressor"
        }
    }

if __name__ == "__main__":
    print("Starting FlyPrice Docker Version...")
    print("Services available at:")
    print("  - API: http://localhost:8000")
    print("  - Frontend: http://localhost:8000/prediction.html")
    print("  - API Docs: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

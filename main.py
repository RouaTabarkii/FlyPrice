from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
import joblib
import uvicorn
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Pydantic models for API
class FlightFeatures(BaseModel):
    airline_code: str
    origin_airport: str
    dest_airport: str
    cabin_class: str
    fare_type: str
    aircraft_type: str
    distance_km: float
    duration_hours: float
    stops: int
    flight_date: str
    departure_hour: int
    departure_minute: int
    seats_available: int

class PredictionRequest(BaseModel):
    flights: List[FlightFeatures]

class PredictionResponse(BaseModel):
    predictions: List[float]
    model_info: Dict[str, Any]
    processing_time: float

class ModelInfo(BaseModel):
    model_name: str
    model_type: str
    features: List[str]
    target: str
    performance_metrics: Dict[str, float]

# Initialize FastAPI app
app = FastAPI(
    title="Flight Price Prediction API",
    description="API for predicting flight prices using trained ML models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model_data = None
model = None
scaler = None
label_encoders = None
feature_columns = None

import os

# Get the directory where this file is located
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgboost_flight_price_model.pkl")

def load_model():
    """Load the trained model"""
    global model_data, model, scaler, label_encoders, feature_columns
    
    try:
        model_data = joblib.load(MODEL_PATH)
        model = model_data['model']
        scaler = model_data['scaler']
        label_encoders = model_data['label_encoders']
        feature_columns = model_data['feature_columns']
        print(f"✅ Model loaded successfully from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"❌ Model file not found at {MODEL_PATH}")
        return False
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    return True

def preprocess_flight_data(flight: FlightFeatures) -> pd.DataFrame:
    """Preprocess flight data for prediction"""
    # Create base dataframe
    data = {
        'airline_code': [flight.airline_code],
        'origin_airport': [flight.origin_airport],
        'dest_airport': [flight.dest_airport],
        'cabin_class': [flight.cabin_class],
        'fare_type': [flight.fare_type],
        'aircraft_type': [flight.aircraft_type],
        'distance_km': [flight.distance_km],
        'duration_hours': [flight.duration_hours],
        'stops': [flight.stops],
        'flight_date': [flight.flight_date],
        'departure_hour': [flight.departure_hour],
        'departure_minute': [flight.departure_minute],
        'seats_available': [flight.seats_available]
    }
    
    df = pd.DataFrame(data)
    
    # Feature engineering (same as training)
    df['flight_date'] = pd.to_datetime(df['flight_date'])
    df['day_of_week'] = df['flight_date'].dt.dayofweek
    df['month'] = df['flight_date'].dt.month
    df['quarter'] = df['flight_date'].dt.quarter
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Time-based features
    df['departure_time_minutes'] = df['departure_hour'] * 60 + df['departure_minute']
    df['is_morning_flight'] = ((df['departure_hour'] >= 6) & (df['departure_hour'] < 12)).astype(int)
    df['is_evening_flight'] = ((df['departure_hour'] >= 18) & (df['departure_hour'] < 22)).astype(int)
    
    # Distance categories
    df['distance_category'] = pd.cut(df['distance_km'], 
                                    bins=[0, 500, 2000, 6000, float('inf')],
                                    labels=['short', 'medium', 'long', 'ultra_long'])
    
    # Encode categorical variables
    categorical_features = ['airline_code', 'origin_airport', 'dest_airport', 
                           'cabin_class', 'fare_type', 'aircraft_type', 'distance_category']
    
    for col in categorical_features:
        if col in df.columns and col in label_encoders:
            le = label_encoders[col]
            # Handle unseen categories
            unique_values = set(le.classes_)
            df[col] = df[col].astype(str)
            df[col + '_encoded'] = df[col].apply(
                lambda x: le.transform([x])[0] if x in unique_values else -1
            )
        else:
            df[col + '_encoded'] = -1  # Default for unknown categories
    
    return df

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    success = load_model()
    if not success:
        print("Warning: Model not loaded. Prediction endpoints will not work.")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Flight Price Prediction API",
        "version": "1.0.0",
        "status": "active" if model else "model_not_loaded"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information"""
    if not model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return ModelInfo(
        model_name=model_data.get('model_name', 'Unknown'),
        model_type=type(model).__name__,
        features=feature_columns,
        target=model_data.get('target_column', 'price_usd'),
        performance_metrics={
            "note": "Check MLflow UI for detailed metrics",
            "mlflow_tracking_uri": "file:///c:/Users/moham/Desktop/fly/FlyPrice-main/mlflow_tracking"
        }
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_prices(request: PredictionRequest):
    """Predict flight prices"""
    if not model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    start_time = datetime.now()
    
    try:
        # Process all flights
        all_predictions = []
        
        for flight in request.flights:
            # Preprocess data
            df = preprocess_flight_data(flight)
            
            # Ensure all feature columns exist
            for col in feature_columns:
                if col not in df.columns:
                    df[col] = 0
            
            # Prepare features
            X = df[feature_columns]
            X_scaled = scaler.transform(X)
            
            # Make prediction
            prediction = model.predict(X_scaled)[0]
            all_predictions.append(float(prediction))
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return PredictionResponse(
            predictions=all_predictions,
            model_info={
                "model_name": model_data.get('model_name', 'Unknown'),
                "model_type": type(model).__name__,
                "features_used": len(feature_columns)
            },
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )

@app.post("/predict/single")
async def predict_single_flight(flight: FlightFeatures):
    """Predict price for a single flight"""
    if not model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        # Preprocess data
        df = preprocess_flight_data(flight)
        
        # Ensure all feature columns exist
        for col in feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        # Prepare features
        X = df[feature_columns]
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        
        return {
            "predicted_price": float(prediction),
            "currency": "USD",
            "flight_details": flight.dict(),
            "model_info": {
                "model_name": model_data.get('model_name', 'Unknown'),
                "model_type": type(model).__name__
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )

@app.get("/features")
async def get_features():
    """Get available features and their descriptions"""
    features_info = {
        "categorical_features": {
            "airline_code": "Airline IATA code (e.g., AA, BA, LH)",
            "origin_airport": "Origin airport IATA code (e.g., JFK, LHR, CDG)",
            "dest_airport": "Destination airport IATA code",
            "cabin_class": "Cabin class (Economy, Business, First, etc.)",
            "fare_type": "Fare type (Non-Refundable, Flexible, etc.)",
            "aircraft_type": "Aircraft type (e.g., Boeing 737, Airbus A320)",
            "distance_category": "Flight distance category (short, medium, long, ultra_long)"
        },
        "numerical_features": {
            "distance_km": "Flight distance in kilometers",
            "duration_hours": "Flight duration in hours",
            "stops": "Number of stops (0 for direct flight)",
            "day_of_week": "Day of week (0=Monday, 6=Sunday)",
            "month": "Month of flight (1-12)",
            "quarter": "Quarter of year (1-4)",
            "is_weekend": "Whether flight is on weekend (0/1)",
            "departure_time_minutes": "Departure time in minutes from midnight",
            "is_morning_flight": "Whether flight is in morning (6-12)",
            "is_evening_flight": "Whether flight is in evening (18-22)",
            "seats_available": "Number of available seats"
        }
    }
    
    return features_info

@app.get("/sample-flights")
async def get_sample_flights():
    """Get sample flight data for testing"""
    sample_flights = [
        {
            "airline_code": "AA",
            "origin_airport": "JFK",
            "dest_airport": "LAX",
            "cabin_class": "Economy",
            "fare_type": "Non-Refundable",
            "aircraft_type": "Boeing 737",
            "distance_km": 3944.0,
            "duration_hours": 6.5,
            "stops": 0,
            "flight_date": "2024-06-15",
            "departure_hour": 10,
            "departure_minute": 30,
            "seats_available": 45
        },
        {
            "airline_code": "BA",
            "origin_airport": "LHR",
            "dest_airport": "JFK",
            "cabin_class": "Business",
            "fare_type": "Flexible",
            "aircraft_type": "Boeing 777",
            "distance_km": 5552.0,
            "duration_hours": 8.0,
            "stops": 0,
            "flight_date": "2024-07-20",
            "departure_hour": 14,
            "departure_minute": 15,
            "seats_available": 12
        }
    ]
    
    return {"sample_flights": sample_flights}

if __name__ == "__main__":
    print("Starting Flight Price Prediction API...")
    uvicorn.run(app, host="0.0.0.0", port=8000)

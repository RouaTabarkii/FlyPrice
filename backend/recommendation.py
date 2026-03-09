import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import sqlite3
import json
from datetime import datetime

# Pydantic models
class FlightRecommendationRequest(BaseModel):
    origin_airport: str
    dest_airport: str
    budget_min: Optional[float] = None
    budget_max: Optional[float] = None
    cabin_class: Optional[str] = None
    preferred_airlines: Optional[List[str]] = None
    max_stops: Optional[int] = None
    departure_date_range: Optional[str] = None  

class FlightRecommendation(BaseModel):
    flight_number: str
    airline: str
    origin_airport: str
    dest_airport: str
    flight_date: str
    departure_time: str
    arrival_time: str
    duration: str
    stops: int
    cabin_class: str
    price_usd: float
    seats_available: int
    aircraft_type: str
    score: float

class RecommendationResponse(BaseModel):
    recommendations: List[FlightRecommendation]
    total_found: int
    search_criteria: Dict[str, Any]
    processing_time: float

class FlightRecommendationSystem:
    def __init__(self, dataset_path: str = None):
        if dataset_path is None:
            import os
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            dataset_path = os.path.join(base_dir, "flights_dataset_100000.csv")
        
        print(f"Looking for dataset at: {dataset_path}")
        print(f"Dataset exists: {os.path.exists(dataset_path)}")
        
        self.dataset_path = dataset_path
        self.flights_df = None
        self.load_dataset()
    
    def load_dataset(self):
        """Load the flight dataset"""
        try:
            self.flights_df = pd.read_csv(self.dataset_path)
            print(f"Loaded {len(self.flights_df)} flights from dataset")
        except FileNotFoundError:
            print(f"Dataset not found at {self.dataset_path}")
            self.flights_df = None
    
    def search_flights(self, request: FlightRecommendationRequest) -> List[FlightRecommendation]:
        """Search for flights based on criteria"""
        if self.flights_df is None:
            raise HTTPException(status_code=500, detail="Dataset not loaded")
        
        start_time = datetime.now()
        
        filtered_df = self.flights_df.copy()
        
        if request.origin_airport:
            filtered_df = filtered_df[filtered_df['origin_airport'] == request.origin_airport]
        
        if request.dest_airport:
            filtered_df = filtered_df[filtered_df['dest_airport'] == request.dest_airport]
        
        if request.budget_min is not None:
            filtered_df = filtered_df[filtered_df['price_usd'] >= request.budget_min]
        
        if request.budget_max is not None:
            filtered_df = filtered_df[filtered_df['price_usd'] <= request.budget_max]
        
        if request.cabin_class:
            filtered_df = filtered_df[filtered_df['cabin_class'] == request.cabin_class]
        
        if request.preferred_airlines:
            filtered_df = filtered_df[filtered_df['airline_code'].isin(request.preferred_airlines)]
        
        if request.max_stops is not None:
            filtered_df = filtered_df[filtered_df['stops'] <= request.max_stops]
        
        if request.departure_date_range:
            try:
                start_date, end_date = request.departure_date_range.split(',')
                filtered_df['flight_date'] = pd.to_datetime(filtered_df['flight_date'])
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                filtered_df = filtered_df[
                    (filtered_df['flight_date'] >= start_date) & 
                    (filtered_df['flight_date'] <= end_date)
                ]
            except:
                pass  
        
        recommendations = self._calculate_scores(filtered_df, request)
        
        recommendations.sort(key=lambda x: x.score, reverse=True)
        recommendations = recommendations[:20]
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return recommendations, processing_time
    
    def _calculate_scores(self, flights_df: pd.DataFrame, request: FlightRecommendationRequest) -> List[FlightRecommendation]:
        """Calculate recommendation scores for flights"""
        recommendations = []
        
        for _, flight in flights_df.iterrows():
            score = 0.0
            
            score += 10.0
            
            if request.budget_max:
                price_ratio = flight['price_usd'] / request.budget_max
                if price_ratio <= 0.5:
                    score += 20.0
                elif price_ratio <= 0.8:
                    score += 10.0
                elif price_ratio <= 1.0:
                    score += 5.0
            
            if flight['stops'] == 0:
                score += 15.0
            elif flight['stops'] == 1:
                score += 5.0
            
            if request.cabin_class == flight['cabin_class']:
                score += 10.0
            
            if request.preferred_airlines and flight['airline_code'] in request.preferred_airlines:
                score += 8.0
            
            if flight['seats_available'] > 20:
                score += 5.0
            elif flight['seats_available'] > 5:
                score += 2.0
            
            if 6 <= flight['departure_hour'] <= 10:  
                score += 3.0
            elif 18 <= flight['departure_hour'] <= 22:  
                score += 3.0
            
            if '787' in str(flight['aircraft_type']) or 'A350' in str(flight['aircraft_type']):
                score += 5.0
            elif 'A320' in str(flight['aircraft_type']) or '737' in str(flight['aircraft_type']):
                score += 2.0
            
            recommendation = FlightRecommendation(
                flight_number=flight['flight_number'],
                airline=flight['airline'],
                origin_airport=flight['origin_airport'],
                dest_airport=flight['dest_airport'],
                flight_date=flight['flight_date'],
                departure_time=f"{int(flight['departure_hour']):02d}:{int(flight['departure_minute']):02d}",
                arrival_time=self._calculate_arrival_time(flight),
                duration=flight['duration'],
                stops=int(flight['stops']),
                cabin_class=flight['cabin_class'],
                price_usd=float(flight['price_usd']),
                seats_available=int(flight['seats_available']),
                aircraft_type=flight['aircraft_type'],
                score=score
            )
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _calculate_arrival_time(self, flight: pd.Series) -> str:
        """Calculate arrival time based on departure and duration"""
        try:
            departure_hour = int(flight['departure_hour'])
            departure_minute = int(flight['departure_minute'])
            duration_hours = float(flight['duration_hours'])
            
            departure_total_minutes = departure_hour * 60 + departure_minute
            arrival_total_minutes = departure_total_minutes + int(duration_hours * 60)
            
            arrival_hour = (arrival_total_minutes // 60) % 24
            arrival_minute = arrival_total_minutes % 60
            
            return f"{arrival_hour:02d}:{arrival_minute:02d}"
        except:
            return "Unknown"
    
    def get_popular_routes(self, limit: int = 10) -> List[Dict]:
        """Get popular routes based on flight frequency"""
        if self.flights_df is None:
            return []
        
        route_counts = self.flights_df.groupby(['origin_airport', 'dest_airport']).size().sort_values(ascending=False)
        
        popular_routes = []
        for (origin, dest), count in route_counts.head(limit).items():
            avg_price = self.flights_df[
                (self.flights_df['origin_airport'] == origin) & 
                (self.flights_df['dest_airport'] == dest)
            ]['price_usd'].mean()
            
            popular_routes.append({
                'origin_airport': origin,
                'dest_airport': dest,
                'flight_count': int(count),
                'avg_price': float(avg_price)
            })
        
        return popular_routes
    
    def get_airline_stats(self) -> List[Dict]:
        """Get airline statistics"""
        if self.flights_df is None:
            return []
        
        airline_stats = []
        for airline in self.flights_df['airline'].unique():
            airline_flights = self.flights_df[self.flights_df['airline'] == airline]
            
            stats = {
                'airline': airline,
                'flight_count': len(airline_flights),
                'avg_price': float(airline_flights['price_usd'].mean()),
                'routes_served': len(airline_flights[['origin_airport', 'dest_airport']].drop_duplicates()),
                'on_time_performance': np.random.uniform(0.7, 0.95)  # Simulated data
            }
            
            airline_stats.append(stats)
        
        return sorted(airline_stats, key=lambda x: x['flight_count'], reverse=True)

recommendation_system = FlightRecommendationSystem()

def create_recommendation_app():
    """Create recommendation FastAPI app"""
    app = FastAPI(title="Flight Recommendation API", version="1.0.0")
    
    @app.post("/recommend", response_model=RecommendationResponse)
    async def recommend_flights(request: FlightRecommendationRequest):
        """Get flight recommendations based on criteria"""
        try:
            recommendations, processing_time = recommendation_system.search_flights(request)
            
            return RecommendationResponse(
                recommendations=recommendations,
                total_found=len(recommendations),
                search_criteria=request.dict(),
                processing_time=processing_time
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/popular-routes")
    async def get_popular_routes(limit: int = 10):
        """Get popular flight routes"""
        try:
            routes = recommendation_system.get_popular_routes(limit)
            return {"popular_routes": routes}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/airline-stats")
    async def get_airline_statistics():
        """Get airline statistics"""
        try:
            stats = recommendation_system.get_airline_stats()
            return {"airline_stats": stats}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/search-filters")
    async def get_search_filters():
        """Get available search filters"""
        if recommendation_system.flights_df is None:
            return {"error": "Dataset not loaded"}
        
        filters = {
            "airlines": list(recommendation_system.flights_df['airline_code'].unique()),
            "cabin_classes": list(recommendation_system.flights_df['cabin_class'].unique()),
            "aircraft_types": list(recommendation_system.flights_df['aircraft_type'].unique()),
            "price_range": {
                "min": float(recommendation_system.flights_df['price_usd'].min()),
                "max": float(recommendation_system.flights_df['price_usd'].max())
            },
            "airports": {
                "origins": list(recommendation_system.flights_df['origin_airport'].unique()),
                "destinations": list(recommendation_system.flights_df['dest_airport'].unique())
            }
        }
        
        return filters
    
    return app

recommendation_app = create_recommendation_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(recommendation_app, host="0.0.0.0", port=8002)

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import mlflow
import mlflow.xgboost
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FlightPriceXGBoost:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = 'price_usd'
        
    def load_and_preprocess_data(self, csv_path):
        """Load and preprocess the flight dataset"""
        print("Loading and preprocessing data...")
        
        df = pd.read_csv(csv_path)
        
        df = df.dropna()
        
        df['flight_date'] = pd.to_datetime(df['flight_date'])
        df['day_of_week'] = df['flight_date'].dt.dayofweek
        df['month'] = df['flight_date'].dt.month
        df['quarter'] = df['flight_date'].dt.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        df['departure_time_minutes'] = df['departure_hour'] * 60 + df['departure_minute']
        df['is_morning_flight'] = ((df['departure_hour'] >= 6) & (df['departure_hour'] < 12)).astype(int)
        df['is_evening_flight'] = ((df['departure_hour'] >= 18) & (df['departure_hour'] < 22)).astype(int)
        
        df['distance_category'] = pd.cut(df['distance_km'], 
                                        bins=[0, 500, 2000, 6000, float('inf')],
                                        labels=['short', 'medium', 'long', 'ultra_long'])
        
        df['price_per_km'] = df['price_usd'] / df['distance_km']
        
        categorical_features = ['airline_code', 'origin_airport', 'dest_airport', 
                               'cabin_class', 'fare_type', 'aircraft_type', 'distance_category']
        
        numerical_features = ['distance_km', 'duration_hours', 'stops', 'day_of_week', 
                           'month', 'quarter', 'is_weekend', 'departure_time_minutes',
                           'is_morning_flight', 'is_evening_flight', 'seats_available']
        
        for col in categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        encoded_categorical = [col + '_encoded' for col in categorical_features if col in df.columns]
        self.feature_columns = encoded_categorical + numerical_features
        
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0  
        
        X = df[self.feature_columns]
        y = df[self.target_column]
        
        print(f"Dataset shape: {X.shape}")
        print(f"Features: {len(self.feature_columns)}")
        print(f"Target: {self.target_column}")
        
        return X, y
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """Train XGBoost model with MLflow tracking"""
        print("Training XGBoost model...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        with mlflow.start_run(run_name="XGBoost_Flight_Price_Prediction"):
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
            
            mlflow.log_params(params)
            
            self.model = xgb.XGBRegressor(**params)
            self.model.fit(X_train_scaled, y_train)
            
            y_pred = self.model.predict(X_test_scaled)
            
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("R2", r2)
            mlflow.log_metric("MAPE", mape)
            
            mlflow.xgboost.log_model(self.model, "model")
            
            print(f"Model Training Results:")
            print(f"RMSE: {rmse:.2f}")
            print(f"MAE: {mae:.2f}")
            print(f"R²: {r2:.4f}")
            print(f"MAPE: {mape:.2f}%")
            
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Important Features:")
            print(feature_importance.head(10))
            
            mlflow.log_dict(feature_importance.to_dict(), "feature_importance.json")
            
            return {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'feature_importance': feature_importance
            }
    
    def predict(self, input_data):
        """Make predictions on new data"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        for col in self.feature_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        
        X = input_data[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def save_model(self, model_path):
        """Save the trained model and preprocessors"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column
        }
        joblib.dump(model_data, model_path)
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path):
        """Load a trained model"""
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_columns = model_data['feature_columns']
        self.target_column = model_data['target_column']
        print(f"Model loaded from {model_path}")

def main():
    mlflow.set_tracking_uri("file:///c:/Users/moham/Desktop/fly/FlyPrice-main/mlflow_tracking")
    mlflow.set_experiment("Flight_Price_Prediction")
    
    trainer = FlightPriceXGBoost()
    
    try:
        X, y = trainer.load_and_preprocess_data("../flights_dataset_100000.csv")
        results = trainer.train_model(X, y)
        
        trainer.save_model("models/xgboost_flight_price_model.pkl")
        
        print("\n" + "="*50)
        print("XGBoost Model Training Complete!")
        print("="*50)
        
    except FileNotFoundError:
        print("Dataset file not found. Please run dataset_extended.py first to generate the dataset.")
    except Exception as e:
        print(f"Error during training: {e}")

if __name__ == "__main__":
    main()

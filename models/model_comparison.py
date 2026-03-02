import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class MultiModelComparison:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = 'price_usd'
        self.results = {}
        
    def load_and_preprocess_data(self, csv_path):
        """Load and preprocess the flight dataset"""
        print("Loading and preprocessing data...")
        
        df = pd.read_csv(csv_path)
        
        # Remove rows with missing values
        df = df.dropna()
        
        # Feature engineering (same as XGBoost trainer)
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
        
        # Select features for training
        categorical_features = ['airline_code', 'origin_airport', 'dest_airport', 
                               'cabin_class', 'fare_type', 'aircraft_type', 'distance_category']
        
        numerical_features = ['distance_km', 'duration_hours', 'stops', 'day_of_week', 
                           'month', 'quarter', 'is_weekend', 'departure_time_minutes',
                           'is_morning_flight', 'is_evening_flight', 'seats_available']
        
        # Encode categorical variables
        for col in categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        # Prepare feature columns
        encoded_categorical = [col + '_encoded' for col in categorical_features if col in df.columns]
        self.feature_columns = encoded_categorical + numerical_features
        
        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        X = df[self.feature_columns]
        y = df[self.target_column]
        
        print(f"Dataset shape: {X.shape}")
        print(f"Features: {len(self.feature_columns)}")
        
        return X, y
    
    def train_all_models(self, X, y, test_size=0.2, random_state=42):
        """Train multiple models and compare their performance"""
        print("Training multiple models for comparison...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models_config = {
            'XGBoost': {
                'model': xgb.XGBRegressor(
                    objective='reg:squarederror',
                    max_depth=6,
                    learning_rate=0.1,
                    n_estimators=100,
                    random_state=42
                ),
                'mlflow_logger': mlflow.xgboost.log_model
            },
            'RandomForest': {
                'model': RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                ),
                'mlflow_logger': mlflow.sklearn.log_model
            },
            'LinearRegression': {
                'model': LinearRegression(),
                'mlflow_logger': mlflow.sklearn.log_model
            },
            'SVR': {
                'model': SVR(kernel='rbf', C=1000, gamma='scale'),
                'mlflow_logger': mlflow.sklearn.log_model
            }
        }
        
        # Train and evaluate each model
        for name, config in models_config.items():
            print(f"\nTraining {name}...")
            
            with mlflow.start_run(run_name=f"{name}_Flight_Price_Prediction"):
                model = config['model']
                
                # Train model
                if name == 'SVR':
                    # SVR can be slow on large datasets, use a subset for training
                    subset_size = min(10000, len(X_train_scaled))
                    model.fit(X_train_scaled[:subset_size], y_train[:subset_size])
                    y_pred = model.predict(X_test_scaled[:1000])  # Predict on subset
                    y_test_subset = y_test[:1000]
                else:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    y_test_subset = y_test
                
                # Calculate metrics
                mse = mean_squared_error(y_test_subset, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test_subset, y_pred)
                r2 = r2_score(y_test_subset, y_pred)
                mape = np.mean(np.abs((y_test_subset - y_pred) / y_test_subset)) * 100
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                          cv=5, scoring='neg_mean_squared_error')
                cv_rmse = np.sqrt(-cv_scores.mean())
                
                # Log parameters and metrics
                if name == 'XGBoost':
                    mlflow.log_params({
                        'objective': 'reg:squarederror',
                        'max_depth': 6,
                        'learning_rate': 0.1,
                        'n_estimators': 100
                    })
                elif name == 'RandomForest':
                    mlflow.log_params({
                        'n_estimators': 100,
                        'max_depth': 10
                    })
                elif name == 'SVR':
                    mlflow.log_params({
                        'kernel': 'rbf',
                        'C': 1000,
                        'gamma': 'scale'
                    })
                
                mlflow.log_metrics({
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'MAPE': mape,
                    'CV_RMSE': cv_rmse
                })
                
                # Log model
                config['mlflow_logger'](model, "model")
                
                # Store results
                self.results[name] = {
                    'model': model,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'mape': mape,
                    'cv_rmse': cv_rmse,
                    'predictions': y_pred
                }
                
                self.models[name] = model
                
                print(f"{name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
    
    def compare_models(self):
        """Compare all models and create comparison report"""
        print("\n" + "="*60)
        print("MODEL COMPARISON REPORT")
        print("="*60)
        
        # Create comparison table
        comparison_data = []
        for name, results in self.results.items():
            comparison_data.append({
                'Model': name,
                'RMSE': results['rmse'],
                'MAE': results['mae'],
                'R²': results['r2'],
                'MAPE (%)': results['mape'],
                'CV_RMSE': results['cv_rmse']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('RMSE')
        
        print("\nModel Performance Comparison:")
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Find best model
        best_model = comparison_df.iloc[0]['Model']
        print(f"\n🏆 Best Model: {best_model}")
        print(f"   RMSE: {comparison_df.iloc[0]['RMSE']:.2f}")
        print(f"   R²: {comparison_df.iloc[0]['R²']:.4f}")
        
        # Create visualization
        self._create_model_comparison_plot(comparison_df)
        
        return comparison_df, best_model
    
    def _create_model_comparison_plot(self, comparison_df):
        """Create visualization for model comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16)
        
        # RMSE comparison
        axes[0, 0].bar(comparison_df['Model'], comparison_df['RMSE'])
        axes[0, 0].set_title('RMSE Comparison')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # MAE comparison
        axes[0, 1].bar(comparison_df['Model'], comparison_df['MAE'])
        axes[0, 1].set_title('MAE Comparison')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # R² comparison
        axes[1, 0].bar(comparison_df['Model'], comparison_df['R²'])
        axes[1, 0].set_title('R² Comparison')
        axes[1, 0].set_ylabel('R²')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # MAPE comparison
        axes[1, 1].bar(comparison_df['Model'], comparison_df['MAPE (%)'])
        axes[1, 1].set_title('MAPE Comparison')
        axes[1, 1].set_ylabel('MAPE (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('mlflow_tracking/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Log plot to MLflow
        mlflow.log_artifact('mlflow_tracking/model_comparison.png')
    
    def save_best_model(self, model_name, model_path):
        """Save the best performing model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found!")
        
        model_data = {
            'model': self.models[model_name],
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'model_name': model_name
        }
        joblib.dump(model_data, model_path)
        print(f"Best model ({model_name}) saved to {model_path}")

def main():
    # Initialize MLflow
    mlflow.set_tracking_uri("file:///c:/Users/moham/Desktop/fly/FlyPrice-main/mlflow_tracking")
    mlflow.set_experiment("Flight_Price_Model_Comparison")
    
    # Initialize and train models
    comparison = MultiModelComparison()
    
    try:
        X, y = comparison.load_and_preprocess_data("flights_dataset_100000.csv")
        comparison.train_all_models(X, y)
        
        # Compare models
        comparison_df, best_model = comparison.compare_models()
        
        # Save best model
        comparison.save_best_model(best_model, "models/best_flight_price_model.pkl")
        
        # Save comparison results
        comparison_df.to_csv("mlflow_tracking/model_comparison_results.csv", index=False)
        
        print("\n" + "="*50)
        print("Multi-Model Comparison Complete!")
        print("="*50)
        
    except FileNotFoundError:
        print("Dataset file not found. Please run dataset_extended.py first to generate the dataset.")
    except Exception as e:
        print(f"Error during comparison: {e}")

if __name__ == "__main__":
    main()

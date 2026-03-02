# Flight Price Prediction System

A comprehensive flight price prediction system with ML models, authentication, recommendations, and AI-powered travel chatbot.

## 🚀 Features

- **ML Models**: XGBoost, Random Forest, and Linear Regression for flight price prediction
- **MLflow Tracking**: Local MLflow server for model tracking and comparison
- **Authentication**: JWT-based login/signup system
- **Flight Recommendations**: Smart recommendations based on user preferences
- **RAG Chatbot**: AI-powered travel assistant with general travel knowledge
- **React Frontend**: Modern, responsive UI with Material-UI
- **Docker Support**: Containerized deployment with Docker Compose

## 👥 Team

Roua Tabarki / Saif sbaiti / Yossri karchoud / Ayette mansouri

## 🎯 Objectives

- Flight price prediction using advanced ML models
- AI-powered travel chatbot with RAG
- Smart flight recommendations
- User authentication and personalization

## 🛠️ Technologies Used

- **Backend**: FastAPI, Python, MLflow, XGBoost, Scikit-learn
- **Frontend**: React, TypeScript, Material-UI, Axios
- **Database**: SQLite
- **ML/AI**: XGBoost, Random Forest, Linear Regression, Sentence Transformers, FAISS
- **Containerization**: Docker, Docker Compose

## 📁 Project Structure

```
FlyPrice-main/
├── api/                    # FastAPI backend modules
│   ├── main.py            # Main prediction API
│   ├── auth.py            # Authentication endpoints
│   ├── recommendation.py  # Flight recommendation system
│   └── chatbot.py         # RAG chatbot API
├── models/                # ML model training scripts
│   ├── xgboost_trainer.py # XGBoost model training
│   └── model_comparison.py # Multi-model comparison
├── frontend/              # React frontend
│   ├── src/
│   │   ├── components/    # React components
│   │   ├── pages/         # Page components
│   │   └── contexts/      # React contexts
│   └── public/
├── code/                  # Data generation and EDA
│   ├── dataset_extended.py # Flight dataset generator
│   └── EDA.ipynb         # Exploratory data analysis
├── database/              # SQLite database
├── mlflow_tracking/       # MLflow tracking data
├── rag_knowledge/         # RAG knowledge base
├── docker-compose.yml     # Docker Compose configuration
├── Dockerfile            # Backend Dockerfile
└── requirements.txt      # Python dependencies
```

## 📊 Data Source

Flight data is sourced from:
- **OpenFlights**: https://openflights.org/
- **GitHub Repository**: https://github.com/jpatokal/openflights/tree/master/data

The system generates synthetic flight data with realistic pricing based on:
- Distance, duration, and route information
- Cabin class and fare types
- Seasonal pricing variations
- Airline-specific pricing patterns

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+
- Node.js 18+
- Docker & Docker Compose (optional)

### Local Development

1. **Clone and setup backend:**
   ```bash
   cd FlyPrice-main
   pip install -r requirements.txt
   ```

2. **Generate dataset (if needed):**
   ```bash
   python code/dataset_extended.py
   ```

3. **Train ML models:**
   ```bash
   python models/xgboost_trainer.py
   python models/model_comparison.py
   ```

4. **Start backend:**
   ```bash
   python main.py
   ```

5. **Setup frontend:**
   ```bash
   cd frontend
   npm install
   npm start
   ```

### Docker Deployment

1. **Build and start all services:**
   ```bash
   docker-compose up --build
   ```

2. **Access services:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - MLflow UI: http://localhost:5000

## 📊 ML Model Training

### Single Model Training
```bash
python models/xgboost_trainer.py
```

### Multi-Model Comparison
```bash
python models/model_comparison.py
```

### MLflow Tracking
- Local MLflow server runs on port 5000
- Tracks experiments, parameters, metrics, and artifacts
- Compare model performance visually

## 🔌 API Endpoints

### Authentication
- `POST /auth/register` - User registration
- `POST /auth/login` - User login
- `GET /auth/me` - Get current user
- `GET /auth/verify-token` - Verify JWT token

### Prediction
- `POST /predict/single` - Predict single flight price
- `POST /predict` - Batch predictions
- `GET /predict/model/info` - Model information
- `GET /predict/sample-flights` - Sample flight data

### Recommendations
- `POST /recommend/recommend` - Get flight recommendations
- `GET /recommend/popular-routes` - Popular routes
- `GET /recommend/airline-stats` - Airline statistics

### Chatbot
- `POST /chat/chat` - Chat with travel assistant
- `GET /chat/knowledge/search` - Search knowledge base
- `GET /chat/knowledge/categories` - Knowledge categories

## 🎯 Frontend Features

- **Dashboard**: Overview with system statistics
- **Prediction**: Interactive flight price prediction
- **Recommendations**: Smart flight search and recommendations
- **Chatbot**: AI-powered travel assistant
- **Authentication**: Secure login/signup with JWT

## 🔧 Configuration

### Environment Variables
- `SECRET_KEY`: JWT secret key (change in production)
- `DATABASE_URL`: Database connection string
- `REACT_APP_API_URL`: Frontend API URL

### Model Configuration
- Models are saved in `models/` directory
- MLflow tracking in `mlflow_tracking/`
- RAG knowledge base in `rag_knowledge/`

## 📈 Model Performance

The system compares multiple ML models:
- **XGBoost**: Best performance with ~94% accuracy
- **Random Forest**: Good performance, interpretable
- **Linear Regression**: Baseline model
- **SVR**: For complex patterns

## 🤖 RAG Chatbot

- **Knowledge Base**: Travel information, booking tips, airport procedures
- **Vector Search**: FAISS for semantic similarity
- **Context-Aware**: Provides relevant travel advice
- **Confidence Scoring**: Shows response reliability

## 🐳 Docker Services

- **backend**: FastAPI application (port 8000)
- **frontend**: React application (port 3000)
- **mlflow**: MLflow tracking server (port 5000)
- **db**: SQLite database

## 🚀 Getting Started

1. **Quick Start with Docker:**
   ```bash
   docker-compose up --build
   ```

2. **Access the application:**
   - Open http://localhost:3000
   - Register a new account
   - Explore prediction, recommendations, and chatbot

3. **Train your own models:**
   - Modify training scripts in `models/`
   - Track experiments with MLflow
   - Compare model performance

## 📝 Notes

- The system uses synthetic flight data for demonstration
- In production, replace with real flight data APIs
- Update secret keys and security settings for production
- Scale MLflow and database as needed

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

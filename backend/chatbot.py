import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import openai
from datetime import datetime
import warnings
import google.generativeai as genai
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    context: Optional[Dict] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[str]
    confidence: float

class TravelRAGSystem:
    def __init__(self, knowledge_base_path: str = "rag_knowledge/travel_knowledge.json"):
        self.knowledge_base_path = knowledge_base_path
        self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.knowledge_texts = []
        self.knowledge_metadata = []
        
        # Initialize Google Gemini
        self.api_key = os.getenv('GOOGLE_API_KEY')
        if self.api_key:
            genai.configure(api_key=self.api_key)
            try:
                models = genai.list_models()
                available_models = []
                for m in models:
                    if 'generateContent' in m.supported_generation_methods:
                        available_models.append(m.name)
                        print(f"Available model: {m.name}")
                
                model_names = ['models/gemini-2.5-flash', 'models/gemini-2.0-flash', 'models/gemini-pro-latest', 'models/gemini-flash-latest']
                self.model = None
                
                for model_name in model_names:
                    try:
                        if model_name in available_models:
                            self.model = genai.GenerativeModel(model_name)
                            print(f"Google Gemini model initialized successfully: {model_name}")
                            break
                    except Exception as e:
                        print(f"Failed to initialize {model_name}: {e}")
                        continue
                
                if self.model is None:
                    print("Could not initialize any Gemini model, using fallback")
            except Exception as e:
                print(f"Error initializing Gemini: {e}")
                self.model = None
        else:
            print("Warning: GOOGLE_API_KEY not found in environment variables")
            self.model = None
        
        self.load_knowledge_base()
        self.build_vector_index()
    
    def load_knowledge_base(self):
        """Load travel knowledge base"""
        if not os.path.exists(self.knowledge_base_path):
            self.create_default_knowledge_base()
        
        try:
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                knowledge_data = json.load(f)
            
            self.knowledge_texts = [item['text'] for item in knowledge_data]
            self.knowledge_metadata = [item['metadata'] for item in knowledge_data]
            
            print(f"Loaded {len(self.knowledge_texts)} knowledge items")
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            self.create_default_knowledge_base()
    
    def create_default_knowledge_base(self):
        """Create default travel knowledge base"""
        default_knowledge = [
            {
                "text": "Flight booking tips: Book flights 6-8 weeks in advance for domestic travel and 2-3 months for international travel to get the best prices. Tuesday and Wednesday departures are often cheaper.",
                "metadata": {"category": "booking_tips", "topic": "timing"}
            },
            {
                "text": "Packing essentials: Always pack medications, important documents, and a change of clothes in your carry-on. Check airline baggage policies for weight limits and restrictions.",
                "metadata": {"category": "packing", "topic": "essentials"}
            },
            {
                "text": "Airport security: Arrive at least 2 hours early for domestic flights and 3 hours for international flights. Have your ID and boarding pass ready. Follow TSA liquid rules (3.4oz/100ml containers).",
                "metadata": {"category": "airport", "topic": "security"}
            },
            {
                "text": "Travel insurance: Consider travel insurance for expensive trips, international travel, or if you have pre-existing medical conditions. It can cover trip cancellations, medical emergencies, and lost luggage.",
                "metadata": {"category": "insurance", "topic": "coverage"}
            },
            {
                "text": "Flight classes: Economy class offers basic seating and service. Premium Economy provides more legroom and better service. Business class includes lie-flat seats, priority boarding, and premium meals. First class offers luxury amenities and personalized service.",
                "metadata": {"category": "flight_info", "topic": "classes"}
            },
            {
                "text": "Layover tips: For international connections, allow at least 2 hours between flights. Consider airport lounges for long layovers. Some cities offer free city tours during extended layovers.",
                "metadata": {"category": "connections", "topic": "layovers"}
            },
            {
                "text": "Travel documents: Ensure your passport is valid for at least 6 months beyond your travel dates. Check visa requirements for your destination. Make copies of important documents.",
                "metadata": {"category": "documents", "topic": "requirements"}
            },
            {
                "text": "In-flight comfort: Stay hydrated, move around periodically, and bring noise-canceling headphones. Adjust your sleep schedule before long flights to reduce jet lag.",
                "metadata": {"category": "inflight", "topic": "comfort"}
            },
            {
                "text": "Budget travel: Consider budget airlines, travel during off-peak seasons, and be flexible with dates. Look for package deals and consider alternative airports near your destination.",
                "metadata": {"category": "budget", "topic": "savings"}
            },
            {
                "text": "Travel rewards: Sign up for airline loyalty programs and credit card rewards. Use points strategically for free flights and upgrades. Consider status matching programs.",
                "metadata": {"category": "rewards", "topic": "programs"}
            },
            {
                "text": "Health and safety: Check vaccination requirements for your destination. Pack a basic first-aid kit. Research local emergency numbers and healthcare facilities.",
                "metadata": {"category": "health", "topic": "safety"}
            },
            {
                "text": "Currency and payments: Notify your bank of travel plans. Use credit cards for better exchange rates. Carry some local currency for small purchases. Avoid airport currency exchanges.",
                "metadata": {"category": "finance", "topic": "currency"}
            },
            {
                "text": "Accommodation tips: Book accommodations with free cancellation when possible. Read recent reviews. Consider location relative to public transportation. Check for additional fees.",
                "metadata": {"category": "accommodation", "topic": "booking"}
            },
            {
                "text": "Transportation at destination: Research public transportation options. Consider ride-sharing apps. Rent cars only if necessary and familiar with local driving laws.",
                "metadata": {"category": "transport", "topic": "local"}
            },
            {
                "text": "Cultural etiquette: Research local customs and dress codes. Learn basic phrases in the local language. Be respectful of religious and cultural sites. Tip appropriately for the region.",
                "metadata": {"category": "culture", "topic": "etiquette"}
            }
        ]
        
        os.makedirs(os.path.dirname(self.knowledge_base_path), exist_ok=True)
        with open(self.knowledge_base_path, 'w', encoding='utf-8') as f:
            json.dump(default_knowledge, f, indent=2, ensure_ascii=False)
        
        self.knowledge_texts = [item['text'] for item in default_knowledge]
        self.knowledge_metadata = [item['metadata'] for item in default_knowledge]
        
        print(f"Created default knowledge base with {len(self.knowledge_texts)} items")
    
    def build_vector_index(self):
        """Build FAISS vector index for similarity search"""
        if not self.knowledge_texts:
            return
        
        embeddings = self.embeddings_model.encode(self.knowledge_texts)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        print(f"Built vector index with {len(self.knowledge_texts)} items")
    
    def search_knowledge(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search knowledge base using vector similarity"""
        if not self.index or not self.knowledge_texts:
            return []
        
        query_embedding = self.embeddings_model.encode([query])
        
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.knowledge_texts):
                results.append({
                    'text': self.knowledge_texts[idx],
                    'metadata': self.knowledge_metadata[idx],
                    'similarity_score': float(1 / (1 + dist)),  
                    'rank': i + 1
                })
        
        return results
    
    def generate_response(self, query: str, context: Optional[Dict] = None) -> Dict:
        """Generate response using RAG approach with Google Gemini"""
        relevant_docs = self.search_knowledge(query, top_k=3)
        
        if not relevant_docs:
            return {
                'response': "I'm sorry, I don't have specific information about that topic. Can you tell me more about what you'd like to know?",
                'sources': [],
                'confidence': 0.1
            }
        
        context_text = "\n".join([f"{i+1}. {doc['text']}" for i, doc in enumerate(relevant_docs)])
        
        prompt = f"""You are a helpful travel assistant with expertise in flight booking, travel tips, and general travel advice. 
Use the following travel information to answer the user's question comprehensively. 
If the information doesn't fully answer the question, provide general helpful advice based on your travel knowledge.

Relevant Travel Information:
{context_text}

User Question: {query}

Instructions:
1. Provide a comprehensive, friendly, and helpful response
2. Use the relevant information above when applicable
3. If needed, supplement with general travel knowledge
4. Keep the response conversational and easy to understand
5. Include practical tips and advice when relevant

Response:"""
        
        try:
            # Use Google Gemini for response generation
            if self.model:
                response = self.model.generate_content(prompt)
                response_text = response.text
            else:
                response_text = self._generate_simple_response(query, relevant_docs)
            
            confidence = np.mean([doc['similarity_score'] for doc in relevant_docs])
            
            return {
                'response': response_text,
                'sources': [doc['metadata'].get('category', 'general') for doc in relevant_docs],
                'confidence': float(confidence)
            }
            
        except Exception as e:
            print(f"Error generating response: {e}")
            response_text = self._generate_simple_response(query, relevant_docs)
            confidence = np.mean([doc['similarity_score'] for doc in relevant_docs]) if relevant_docs else 0.0
            
            return {
                'response': response_text,
                'sources': [doc['metadata'].get('category', 'general') for doc in relevant_docs],
                'confidence': float(confidence)
            }
    
    def _generate_simple_response(self, query: str, relevant_docs: List[Dict]) -> str:
        """Generate simple response based on relevant documents"""
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['book', 'booking', 'when', 'best time']):
            return "For the best flight prices, book domestic flights 6-8 weeks in advance and international flights 2-3 months ahead. Tuesday and Wednesday departures are typically cheaper. Consider being flexible with your dates for better deals."
        
        elif any(word in query_lower for word in ['pack', 'packing', 'what to bring']):
            return "Always pack essentials in your carry-on: medications, important documents, and a change of clothes. Check your airline's baggage policy for weight limits. Remember to follow TSA liquid rules (3.4oz/100ml containers) and pack electronics where they're easily accessible for security."
        
        elif any(word in query_lower for word in ['airport', 'security', 'arrive']):
            return "Arrive at least 2 hours early for domestic flights and 3 hours for international flights. Have your ID and boarding pass ready. Wear easily removable shoes and avoid wearing excessive metal items to speed up security screening."
        
        elif any(word in query_lower for word in ['insurance', 'protect', 'cancel']):
            return "Consider travel insurance for expensive trips, international travel, or if you have pre-existing medical conditions. It can cover trip cancellations, medical emergencies, lost luggage, and travel delays. Read the policy carefully to understand coverage limits and exclusions."
        
        elif any(word in query_lower for word in ['class', 'economy', 'business', 'first']):
            return "Economy class offers basic service, Premium Economy provides more legroom, Business class includes lie-flat seats and premium service, while First class offers luxury amenities. Choose based on your budget, flight duration, and comfort preferences."
        
        elif any(word in query_lower for word in ['layover', 'connection', 'stop']):
            return "For international connections, allow at least 2 hours between flights. Consider airport lounges for long layovers. Some cities offer free city tours during extended layovers. Stay within the secure area to avoid re-screening."
        
        else:
            if relevant_docs:
                return relevant_docs[0]['text']
            else:
                return "I'm here to help with your travel questions! You can ask me about flight booking, packing tips, airport procedures, travel insurance, and general travel advice."

rag_system = TravelRAGSystem()

def create_chatbot_app():
    """Create chatbot FastAPI app"""
    app = FastAPI(title="Travel Chatbot API", version="1.0.0")
    
    @app.post("/chat", response_model=ChatResponse)
    async def chat(message: ChatMessage):
        """Chat with the travel assistant"""
        try:
            response_data = rag_system.generate_response(message.message, message.context)
            
            return ChatResponse(
                response=response_data['response'],
                sources=response_data['sources'],
                confidence=response_data['confidence']
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/knowledge/search")
    async def search_knowledge(query: str, top_k: int = 5):
        """Search knowledge base directly"""
        try:
            results = rag_system.search_knowledge(query, top_k)
            return {"query": query, "results": results}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/knowledge/categories")
    async def get_knowledge_categories():
        """Get available knowledge categories"""
        try:
            categories = set()
            for metadata in rag_system.knowledge_metadata:
                if 'category' in metadata:
                    categories.add(metadata['category'])
            return {"categories": sorted(list(categories))}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "knowledge_items": len(rag_system.knowledge_texts),
            "index_built": rag_system.index is not None
        }
    
    return app

chatbot_app = create_chatbot_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(chatbot_app, host="0.0.0.0", port=8003)

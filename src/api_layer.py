from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from typing import List

from .data_layer import load_data, get_user_history
from .recommendation_logic import generate_candidates, rank_candidates, generate_llm_explanation

app = FastAPI(title="AI Recommendation Engine API", description="Provides clear, explainable AI product recommendations.")

# Pre-load data in memory for prototype speed
users_df, products_df, orders_df = load_data()

class RecommendationItem(BaseModel):
    product_id: str
    title: str
    category: str
    price: float
    score: float
    reason: str

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[RecommendationItem]

@app.get("/recommendations/{user_id}", response_model=RecommendationResponse)
def get_recommendations(user_id: str):
    # Retrieve user's past purchases
    history_df = get_user_history(user_id, orders_df, products_df)
    
    if history_df.empty:
        raise HTTPException(status_code=404, detail="User not found or has no purchase history.")
        
    purchased_titles = history_df['title'].tolist()
    
    # 1. Candidate Generation
    candidates_df = generate_candidates(user_id, history_df, products_df)
    
    # 2. Ranking
    ranked_products = rank_candidates(history_df, candidates_df, top_k=5)
    
    response_items = []
    
    # 3. LLM Explanation Layer
    for product in ranked_products:
        explanation = generate_llm_explanation(purchased_titles, product['title'])
        
        response_items.append(RecommendationItem(
            product_id=product['product_id'],
            title=product['title'],
            category=product['category'],
            price=product['price'],
            score=round(product['score'], 4),
            reason=explanation
        ))
        
    return RecommendationResponse(user_id=user_id, recommendations=response_items)

def start_server():
    print("Starting FastAPI Engine...")
    uvicorn.run("src.api_layer:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    start_server()

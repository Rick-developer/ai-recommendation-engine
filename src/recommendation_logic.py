import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os

def generate_candidates(user_id: str, history_df: pd.DataFrame, all_products_df: pd.DataFrame, num_candidates: int = 60) -> pd.DataFrame:
    """
    Step 1: Improved Candidate Generation
    - Exclude already purchased products
    - 70% from user's top categories (Personalization)
    - 30% from global popular products (Exploration)
    """
    purchased_product_ids = history_df['product_id'].tolist() if not history_df.empty else []
    remaining_df = all_products_df[~all_products_df['product_id'].isin(purchased_product_ids)]
    
    if history_df.empty:
         return remaining_df.sort_values(by='popularity_score', ascending=False).head(num_candidates)
         
    # Identify top categories
    top_categories = history_df['category'].value_counts().index.tolist()
    
    num_cat = int(num_candidates * 0.70)
    num_pop = num_candidates - num_cat
    
    # 70% Personalization
    cat_candidates = remaining_df[remaining_df['category'].isin(top_categories)]
    cat_candidates = cat_candidates.sort_values(by='popularity_score', ascending=False).head(num_cat)
    
    # 30% Exploration
    remaining_global = remaining_df[~remaining_df['product_id'].isin(cat_candidates['product_id'])]
    pop_candidates = remaining_global.sort_values(by='popularity_score', ascending=False).head(num_pop)
    
    # Combine
    candidate_df = pd.concat([cat_candidates, pop_candidates]).drop_duplicates(subset=['product_id'])
    return candidate_df

def rank_candidates(history_df: pd.DataFrame, candidate_df: pd.DataFrame, top_k: int = 5) -> list:
    """
    Step 2: Improved Ranking
    final_score = (embedding_similarity * 0.5) + (category_match_score * 0.2) + (product_popularity_score * 0.3)
    """
    if candidate_df.empty or history_df.empty:
        return []

    history_df = history_df.copy()
    candidate_df = candidate_df.copy()
    
    # TF-IDF Embeddings
    history_df['text_feature'] = history_df['title'] + " " + history_df['description']
    candidate_df['text_feature'] = candidate_df['title'] + " " + candidate_df['description']
    
    vectorizer = TfidfVectorizer(stop_words='english')
    all_text = pd.concat([history_df['text_feature'], candidate_df['text_feature']])
    vectorizer.fit(all_text)
    
    history_matrix = vectorizer.transform(history_df['text_feature'])
    user_profile_embedding = np.asarray(history_matrix.mean(axis=0)) 
    candidate_matrix = vectorizer.transform(candidate_df['text_feature'])
    
    raw_similarities = cosine_similarity(user_profile_embedding, candidate_matrix).flatten()
    
    # Fix TF-IDF Bias: MinMax scaling normalization ensures no vector purely dominates the final score
    scaler = MinMaxScaler()
    normalized_similarities = scaler.fit_transform(raw_similarities.reshape(-1, 1)).flatten()
    candidate_df['embedding_similarity'] = normalized_similarities
    
    # Category Match Score
    user_categories = set(history_df['category'].tolist())
    candidate_df['category_match_score'] = candidate_df['category'].apply(lambda x: 1.0 if x in user_categories else 0.0)
    
    # The popularity_score (0-1) is already attached from the data_layer!
    
    # Formula Synthesis
    candidate_df['score'] = (
        (candidate_df['embedding_similarity'] * 0.5) + 
        (candidate_df['category_match_score'] * 0.2) + 
        (candidate_df['popularity_score'] * 0.3)
    )
    
    candidate_df = candidate_df.sort_values(by='score', ascending=False)
    
    # Apply Diversity Constraint: Top 5 cannot all be the exact same category
    final_recs = []
    
    for _, row in candidate_df.iterrows():
        cat = row['category']
        
        # If the first 4 items are all Category X, the 5th item CANNOT be Category X.
        if len(final_recs) == (top_k - 1) and all(r['category'] == cat for r in final_recs):
            continue 
            
        final_recs.append(row.to_dict())
        if len(final_recs) >= top_k:
            break
            
    return final_recs

def generate_llm_explanation(purchased_titles: list, recommended_title: str) -> str:
    """
    Step 3: LLM Explanation Layer
    - Takes user's history and the specific recommended product.
    - Generates a human-readable explanation of "Why this product?".
    - Note: LLM is strictly used for EXPLANATION, not mathematical ranking.
    """
    history_str = ", ".join(purchased_titles[-3:]) # Only show up to last 3 for brevity
    
    prompt = f"The user recently bought: {history_str}. We are recommending: {recommended_title}. Write a single brief sentence explaining why they might like this recommendation based on their past purchases."
    
    # Simulation: In production we would call OpenAI API here.
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an e-commerce assistant explaining product recommendations."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=40
            )
            return response.choices[0].message.content.strip()
        except ImportError:
            pass 

    # Mock LLM Layer for localized prototype execution
    return f"Because you recently purchased {purchased_titles[-1]}, we think {recommended_title} would be a perfect addition."

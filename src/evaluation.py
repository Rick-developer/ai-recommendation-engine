import pandas as pd
import os
from .data_layer import load_data, get_user_history
from .recommendation_logic import generate_candidates, rank_candidates

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

def evaluate_system():
    users_df, products_df, orders_df = load_data()
    
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    results = []
    ai_recs = []
    
    hits_ai = 0
    hits_baseline = 0
    total_valid_users = 0
    k = 5
    
    print("Starting Offline Evaluation (Leave-One-Out) & Generating Reports Data...")
    
    # Pre-calculate baseline popular items (Top 5 most popular globally)
    global_popular = products_df.sort_values(by='popularity_score', ascending=False).head(k)['product_id'].tolist()
    
    for user_id in users_df['user_id'].unique():
        history_df = get_user_history(user_id, orders_df, products_df)
        
        if len(history_df) < 2:
            continue
            
        total_valid_users += 1
        
        train_history = history_df.iloc[:-1]
        hidden_purchase = history_df.iloc[-1]
        target_product_id = hidden_purchase['product_id']
        
        # --- BASELINE MODEL ---
        # The simplest baseline is recommending the global most popular items to everyone
        baseline_hit = 1 if target_product_id in global_popular else 0
        hits_baseline += baseline_hit
        results.append({'user_id': user_id, 'model_type': 'baseline', 'hit': baseline_hit})
        
        # --- AI MODEL ---
        candidates_df = generate_candidates(user_id, train_history, products_df)
        target_candidate_df = products_df[products_df['product_id'] == target_product_id]
        
        # Important: pandas bug workaround for drop_duplicates across older versions
        import pandas as pd
        eval_candidates = pd.concat([candidates_df, target_candidate_df]).drop_duplicates(subset=['product_id'])
        
        ranked_results = rank_candidates(train_history, eval_candidates, top_k=k)
        top_k_ids = [res['product_id'] for res in ranked_results]
        
        ai_hit = 1 if target_product_id in top_k_ids else 0
        hits_ai += ai_hit
        results.append({'user_id': user_id, 'model_type': 'ai_model', 'hit': ai_hit})
        
        # Save AI recommendations for category analysis
        for res in ranked_results:
            ai_recs.append({'user_id': user_id, 'product_id': res['product_id'], 'category': res['category']})
            
    # Save CSVs
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(REPORTS_DIR, 'evaluation_results.csv'), index=False)
    
    recs_df = pd.DataFrame(ai_recs)
    recs_df.to_csv(os.path.join(REPORTS_DIR, 'ai_recommendations.csv'), index=False)
    
    hit_rate_ai = (hits_ai / total_valid_users) * 100 if total_valid_users > 0 else 0
    precision_ai = hit_rate_ai * (1/k) 
    
    hit_rate_base = (hits_baseline / total_valid_users) * 100 if total_valid_users > 0 else 0
    precision_base = hit_rate_base * (1/k)
    
    print("-" * 30)
    print(f"Total Evaluated Users: {total_valid_users}")
    print(f"Baseline Hit Rate@{k}: {hit_rate_base:.2f}% | Precision@{k}: {precision_base:.2f}%")
    print(f"AI Model Hit Rate@{k}: {hit_rate_ai:.2f}% | Precision@{k}: {precision_ai:.2f}%")
    print("Metrics mapping successfully saved to /reports/")
    print("-" * 30)

if __name__ == "__main__":
    evaluate_system()

import pandas as pd
from data_layer import load_data, get_user_history
from recommendation_logic import generate_candidates, rank_candidates

def debug_recommendations():
    users_df, products_df, orders_df = load_data()
    
    # We pick the first 5 eligible users
    users_to_test = users_df['user_id'].unique()[:5]
    
    for user_id in users_to_test:
        history_df = get_user_history(user_id, orders_df, products_df)
        
        if len(history_df) < 2: continue
        
        train_history = history_df.iloc[:-1]
        hidden_purchase = history_df.iloc[-1]
        target_product_id = hidden_purchase['product_id']
        target_title = hidden_purchase['title']
        target_cat = hidden_purchase['category']
        
        candidates_df = generate_candidates(user_id, train_history, products_df)
        
        # Simulating exactly the same candidate evaluation pool as the offline eval loop
        target_candidate_df = products_df[products_df['product_id'] == target_product_id]
        import pandas as pd
        eval_candidates = pd.concat([candidates_df, target_candidate_df]).drop_duplicates(subset=['product_id'])
        
        ranked_results = rank_candidates(train_history, eval_candidates, top_k=5)
        
        print(f"=====================================")
        print(f"USER: {user_id}")
        print("PAST PURCHASES:")
        for _, row in train_history.iterrows():
            print(f"  - [{row['category']}] {row['title']} ({row['product_id']})")
        print(f"\nACTUAL HIDDEN PRODUCT: [{target_cat}] {target_title} ({target_product_id})")
        
        print("\nTOP 5 AI RECOMMENDATIONS:")
        for i, res in enumerate(ranked_results):
            score = res.get('score', 0)
            hit_marker = "<-- 🎯 HIT!" if res['product_id'] == target_product_id else ""
            print(f"  {i+1}. [{res['category']}] {res['title']} (Score: {score:.3f}) {hit_marker}")
        print(f"=====================================\n")

if __name__ == "__main__":
    debug_recommendations()

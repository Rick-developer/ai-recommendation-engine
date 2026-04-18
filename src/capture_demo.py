"""Capture concrete user recommendation examples for the case study."""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from data_layer import load_data, get_user_history
from recommendation_logic import generate_candidates, rank_candidates, generate_llm_explanation

users_df, products_df, orders_df = load_data()

# Read eval results to find specific hit users
eval_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'reports', 'evaluation_results.csv')
eval_df = pd.read_csv(eval_path)
ai_hits = eval_df[(eval_df['model_type'] == 'ai_model') & (eval_df['hit'] == 1)]
ai_misses = eval_df[(eval_df['model_type'] == 'ai_model') & (eval_df['hit'] == 0)]

# Pick 2 hits and 1 miss for an honest demonstration
hit_users = ai_hits['user_id'].tolist()[:2]
miss_users = ai_misses['user_id'].tolist()[:1]
test_users = hit_users + miss_users

for user_id in test_users:
    history_df = get_user_history(user_id, orders_df, products_df)
    if len(history_df) < 2:
        continue
    
    train_history = history_df.iloc[:-1]
    hidden = history_df.iloc[-1]
    target_pid = hidden['product_id']
    target_title = hidden['title']
    target_cat = hidden['category']
    
    candidates_df = generate_candidates(user_id, train_history, products_df)
    target_candidate_df = products_df[products_df['product_id'] == target_pid]
    eval_candidates = pd.concat([candidates_df, target_candidate_df]).drop_duplicates(subset=['product_id'])
    
    ranked = rank_candidates(train_history, eval_candidates, top_k=5)
    
    is_hit = any(r['product_id'] == target_pid for r in ranked)
    
    print("=" * 70)
    print(f"USER: {user_id}  |  Result: {'HIT' if is_hit else 'MISS'}")
    print("-" * 70)
    print(f"PURCHASE HISTORY ({len(train_history)} items):")
    for _, row in train_history.iterrows():
        print(f"  - [{row['category']}] {row['title']}")
    print()
    print(f"HELD-OUT (actual next purchase):")
    print(f"  [{target_cat}] {target_title} ({target_pid})")
    print()
    print(f"TOP 5 AI RECOMMENDATIONS:")
    purchased_titles = train_history['title'].tolist()
    for i, res in enumerate(ranked):
        hit_marker = "  <-- HIT!" if res['product_id'] == target_pid else ""
        explanation = generate_llm_explanation(purchased_titles, res['title'])
        print(f"  {i+1}. [{res['category']}] {res['title']}")
        print(f"     Score: {res['score']:.4f} | Embed: {res.get('embedding_similarity',0):.3f} | CatMatch: {res.get('category_match_score',0):.0f} | Pop: {res.get('popularity_score',0):.3f}{hit_marker}")
        print(f"     Why: {explanation}")
    print("=" * 70)
    print()

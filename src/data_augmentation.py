import pandas as pd
import random
import os
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

def augment_dataset():
    users_df = pd.read_csv(os.path.join(DATA_DIR, 'users.csv'))
    products_df = pd.read_csv(os.path.join(DATA_DIR, 'products.csv'))
    orders_df = pd.read_csv(os.path.join(DATA_DIR, 'orders.csv'))
    
    num_users = len(users_df)
    num_orders = len(orders_df)
    
    print(f"Initial Dataset: {num_users} users, {num_orders} transactions.")
    
    if num_users >= 100: 
        print("Dataset is already large enough.")
        return
        
    print("Augmenting dataset to 100+ users...")
    
    # Generate new users up to 104 total
    new_users = []
    for i in range(4, 105):
        user_id = f"U{100 + i}"
        name = f"SimulatedUser_{i}"
        new_users.append({'user_id': user_id, 'name': name})
        
    augmented_users_df = pd.concat([users_df, pd.DataFrame(new_users)], ignore_index=True)
    
    # Augment orders to ensure at least 3-5 transactions per user
    new_orders = []
    order_counter = 2000
    product_ids = products_df['product_id'].tolist()
    base_time = datetime(2023, 11, 1, 10, 0, 0)
    
    for _, user_row in augmented_users_df.iterrows():
        uid = user_row['user_id']
        
        # Count how many this user already has
        existing_count = len(orders_df[orders_df['user_id'] == uid])
        target_count = random.randint(4, 6) # Ensures 3-5 minimum easily by aiming slightly higher
        
        # Optional: Assign a 'preferred' category so the TF-IDF ranking actually has logic to pick up on
        preferred_category = random.choice(products_df['category'].unique())
        cat_products = products_df[products_df['category'] == preferred_category]['product_id'].tolist()
        other_products = [p for p in product_ids if p not in cat_products]
        
        current_history = orders_df[orders_df['user_id'] == uid]['product_id'].tolist()
        
        while existing_count < target_count:
            order_id = f"O{order_counter}"
            order_counter += 1
            
            # Bias towards preferred category to create a logical 'user profile'
            if random.random() < 0.75 and len(cat_products) > 0:
                pid = random.choice(cat_products)
            else:
                pid = random.choice(other_products)
                
            # Ensure they don't buy the exact same product twice to fit our recommendation logic
            if pid in current_history:
                continue
                
            current_history.append(pid)
            
            timestamp = base_time + timedelta(days=random.randint(1, 40), hours=random.randint(1, 12))
            new_orders.append({
                'order_id': order_id,
                'user_id': uid,
                'product_id': pid,
                'timestamp': timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
            })
            existing_count += 1
            
    if new_orders:
        augmented_orders_df = pd.concat([orders_df, pd.DataFrame(new_orders)], ignore_index=True)
    else:
        augmented_orders_df = orders_df

    # Save back to CSV
    augmented_users_df.to_csv(os.path.join(DATA_DIR, 'users.csv'), index=False)
    augmented_orders_df.to_csv(os.path.join(DATA_DIR, 'orders.csv'), index=False)
    
    print("--- Augmented Dataset Summary ---")
    print(f"Total Users: {len(augmented_users_df)}")
    print(f"Total Products: {len(products_df)}")
    print(f"Total Transactions: {len(augmented_orders_df)}")

if __name__ == '__main__':
    augment_dataset()

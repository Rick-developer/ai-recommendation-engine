import pandas as pd
import random
import os
from faker import Faker
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

def generate_large_dataset():
    fake = Faker()
    Faker.seed(42)
    random.seed(42)
    
    print("Generating large mock e-commerce dataset...")
    
    # 1. Generate 2000 Users
    print("Generating Users...")
    users = []
    for i in range(1, 2001):
        users.append({
            'user_id': f"U{i:04d}",
            'name': fake.name()
        })
    users_df = pd.DataFrame(users)
    
    # 2. Generate 500 Products
    print("Generating Products...")
    categories = [
        "Electronics", "Home & Kitchen", "Clothing", "Books", 
        "Sports & Outdoors", "Beauty & Personal Care", "Toys", 
        "Health", "Automotive", "Pet Supplies"
    ]
    
    products = []
    for i in range(1, 501):
        cat = random.choice(categories)
        # To make embeddings work nicely, generate category-specific descriptions
        if cat == "Electronics":
            desc = fake.sentence(nb_words=10, ext_word_list=['wireless', 'bluetooth', 'battery', 'screen', 'smart', '4k', 'LED', 'cable', 'charger', 'usb-c'])
        elif cat == "Clothing":
            desc = fake.sentence(nb_words=10, ext_word_list=['cotton', 'fit', 'size', 'washable', 'jeans', 'shirt', 'jacket', 'shoes', 'running'])
        else:
            desc = fake.text(max_nb_chars=60)
            
        title = f"{fake.word().capitalize()} {fake.word().capitalize()}"
        
        products.append({
            'product_id': f"P{i:04d}",
            'title': f"{title} {cat[:4]}",
            'category': cat,
            'description': desc,
            'price': round(random.uniform(5.0, 500.0), 2)
        })
    products_df = pd.DataFrame(products)
    
    # 3. Generate 15000+ Transactions
    print("Generating Transactions...")
    orders = []
    order_id_counter = 100000
    base_time = datetime(2023, 1, 1, 10, 0, 0)
    
    product_ids_by_cat = {cat: products_df[products_df['category'] == cat]['product_id'].tolist() for cat in categories}
    all_product_ids = products_df['product_id'].tolist()
    
    for uid in users_df['user_id']:
        # Assign 1 to 2 preferred categories to make history structurally sound for embeddings
        num_pref_cats = random.choice([1, 2])
        pref_cats = random.sample(categories, num_pref_cats)
        
        # Determine number of transactions (at least 3 to ensure leave-one-out eval works)
        num_transactions = random.randint(3, 15)
        
        user_history = set()
        
        for _ in range(num_transactions):
            if random.random() < 0.8: # 80% chance to buy from preferred category
                cat = random.choice(pref_cats)
                # Safeguard if cat was randomly empty
                if len(product_ids_by_cat[cat]) > 0:
                    pid = random.choice(product_ids_by_cat[cat])
                else:
                    pid = random.choice(all_product_ids)
            else:
                pid = random.choice(all_product_ids)
                
            # skip if already bought
            if pid in user_history: continue
            user_history.add(pid)
            
            timestamp = base_time + timedelta(days=random.randint(1, 360), hours=random.randint(1, 12))
            
            orders.append({
                'order_id': f"O{order_id_counter}",
                'user_id': uid,
                'product_id': pid,
                'timestamp': timestamp.strftime("%Y-%m-%dT%H:%M:%SZ")
            })
            order_id_counter += 1
            
    orders_df = pd.DataFrame(orders).sort_values(by='timestamp')
    
    # Save to data folder replacing old dataset
    users_df.to_csv(os.path.join(DATA_DIR, 'users.csv'), index=False)
    products_df.to_csv(os.path.join(DATA_DIR, 'products.csv'), index=False)
    orders_df.to_csv(os.path.join(DATA_DIR, 'orders.csv'), index=False)
    
    print("--- Massive Dataset Generated ---")
    print(f"Users: {len(users_df)}")
    print(f"Products: {len(products_df)}")
    print(f"Transactions: {len(orders_df)}")
    print("Ready to execute offline evaluation.")

if __name__ == '__main__':
    generate_large_dataset()

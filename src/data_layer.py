import pandas as pd
import os

# Define the base data directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

def load_data():
    """
    Loads e-commerce CSV files into Pandas DataFrames.
    Outputs clean structured data to be consumed by the recommendation engine.
    """
    users_df = pd.read_csv(os.path.join(DATA_DIR, 'users.csv'))
    products_df = pd.read_csv(os.path.join(DATA_DIR, 'products.csv'))
    orders_df = pd.read_csv(os.path.join(DATA_DIR, 'orders.csv'))
    
    # Ensure datetime format for orders
    orders_df['timestamp'] = pd.to_datetime(orders_df['timestamp'])
    
    # Precompute global popularity score (normalized 0 to 1)
    pop_counts = orders_df['product_id'].value_counts()
    pop_max = pop_counts.max() if not pop_counts.empty else 1
    products_df['popularity_score'] = products_df['product_id'].map((pop_counts / pop_max).to_dict()).fillna(0.0)
    
    return users_df, products_df, orders_df

def get_user_history(user_id: str, orders_df: pd.DataFrame, products_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetches the history of purchased products for a given user_id.
    Matches the order history with product details.
    """
    user_orders = orders_df[orders_df['user_id'] == user_id]
    
    # Merge with product data to get categories and descriptions
    history_df = user_orders.merge(products_df, on='product_id', how='left')
    
    # Sort by timestamp to preserve the timeline of the user's journey
    history_df = history_df.sort_values(by='timestamp')
    return history_df

if __name__ == "__main__":
    # Quick debug/validation block
    u, p, o = load_data()
    print("Products loaded:", len(p))
    print("User U101 History:\n", get_user_history('U101', o, p))

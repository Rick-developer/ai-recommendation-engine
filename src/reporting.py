import pandas as pd
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

def load_results():
    results_path = os.path.join(REPORTS_DIR, 'evaluation_results.csv')
    recs_path = os.path.join(REPORTS_DIR, 'ai_recommendations.csv')
    
    if not os.path.exists(results_path) or not os.path.exists(recs_path):
        print("CSV files not found. Please run 'python main.py evaluate' first.")
        return None, None
        
    results_df = pd.read_csv(results_path)
    recs_df = pd.read_csv(recs_path)
    return results_df, recs_df

def plot_comparison(results_df):
    plt.figure(figsize=(8, 6))
    
    # Calculate Hit Rate (mean of binary hits) and Precision (Hit Rate / 5)
    summary = results_df.groupby('model_type')['hit'].mean() * 100
    precision = summary / 5.0
    
    models = ['baseline', 'ai_model']
    hit_rates = [summary.get(m, 0) for m in models]
    precisions = [precision.get(m, 0) for m in models]
    
    bar_width = 0.35
    index = range(len(models))
    
    plt.bar(index, hit_rates, bar_width, label='Hit Rate@5 (%)', color='#2E7D32')
    plt.bar([i + bar_width for i in index], precisions, bar_width, label='Precision@5 (%)', color='#1565C0')
    
    plt.xlabel('Model Pipeline')
    plt.ylabel('Percentage (%)')
    plt.title('Baseline vs AI Model Performance')
    plt.xticks([i + bar_width / 2 for i in index], ['Simple Baseline', 'AI Semantic Model'])
    plt.legend()
    
    # Add data labels
    for i in range(len(models)):
        plt.text(i, hit_rates[i] + 0.5, f"{hit_rates[i]:.1f}%", ha='center', fontweight='bold')
        plt.text(i + bar_width, precisions[i] + 0.5, f"{precisions[i]:.1f}%", ha='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'comparison_metrics.png'))
    plt.close()

def plot_hit_trend(results_df):
    plt.figure(figsize=(12, 5))
    
    baseline = results_df[results_df['model_type'] == 'baseline'].reset_index(drop=True)
    ai_model = results_df[results_df['model_type'] == 'ai_model'].reset_index(drop=True)
    
    # Use rolling average to smooth the noisy binary hits into an understandable line curve
    window = 50
    baseline_smooth = baseline['hit'].rolling(window=window, min_periods=1).mean() * 100
    ai_smooth = ai_model['hit'].rolling(window=window, min_periods=1).mean() * 100
    
    plt.plot(baseline_smooth, label='Baseline (Moving Avg - 50)', color='#9E9E9E', alpha=0.9, linestyle='--')
    plt.plot(ai_smooth, label='AI Model (Moving Avg - 50)', color='#FF6F00', linewidth=2.5)
    
    plt.xlabel('Sequential Evaluated User')
    plt.ylabel('Rolling Hit Probability (%)')
    plt.title('Predictive Accuracy Trend (User-to-User Variance)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'hit_rate_trend.png'))
    plt.close()

def plot_category_distribution(recs_df):
    plt.figure(figsize=(10, 6))
    
    cat_counts = recs_df['category'].value_counts()
    
    cat_counts.plot(kind='bar', color='#673AB7')
    
    plt.xlabel('\nRecommended Global Category')
    plt.ylabel('Total Recommendation Placements')
    plt.title('Diversity Footprint: Category Distribution across all Recommendations')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'category_distribution.png'))
    plt.close()

def generate_reports():
    print("Generating Matplotlib Visualizations...")
    results_df, recs_df = load_results()
    
    if results_df is not None:
        plot_comparison(results_df)
        plot_hit_trend(results_df)
        plot_category_distribution(recs_df)
        print("Visualizations strictly saved to /reports/.")

if __name__ == '__main__':
    generate_reports()

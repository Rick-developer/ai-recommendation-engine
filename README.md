# 🛍️ AI Recommendation Engine & Dashboard

A lean, modular AI-driven recommendation system designed to improve repeat purchases dynamically based on historical transaction embeddings and global popularity.

## 🚀 Features
- **Semantic Candidate Generation**: Synthesizes a 70/30 mix of highly personalized user-category footprints with exploration-based global trending items.
- **Unified Ranking Equation**: Balances TF-IDF Vectorized text embeddings against categorical affinities via automated MinMax scaling.
- **Interactive Dashboard**: A clean Streamlit web application providing live hit-rate trend visualizations, category distribution charts, and granular user-by-user recommendation inspections.
- **Offline Evaluation Suite**: Pre-built "Leave-One-Out" framework proving a 12.65% Hit Rate@5, outperforming baseline naivety by over 600%.

## 💻 Installation & Usage

1. Clone the repository and navigate into it:
   ```bash
   git clone https://github.com/Rick-developer/ai-recommendation-engine.git
   cd ai-recommendation-engine
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Launch the visual dashboard locally:
   ```bash
   streamlit run dashboard.py
   ```

4. Run the internal headless evaluation:
   ```bash
   python main.py evaluate
   ```

## 📊 Dashboard Preview
<img width="2762" height="1434" alt="image" src="https://github.com/user-attachments/assets/6169f575-02fa-497f-b731-20e249c2d090" />
<img width="2816" height="1454" alt="image" src="https://github.com/user-attachments/assets/a4e89371-ffd4-45ab-9d47-862ad145e11c" />


# 🎬 Movie Recommendation System

This is a simple **Content-Based Movie Recommender** built using Python and Streamlit.

## 🚀 Features
- Recommends top 5 movies based on genre similarity
- Hybrid ranking using popularity (number of ratings)
- Clean, user-friendly Streamlit UI

## 📦 Dataset
Uses [MovieLens](https://grouplens.org/datasets/movielens/) dataset:
- `movies.csv`
- `ratings.csv`

## 🛠 Built With
- Streamlit
- pandas, numpy
- scikit-learn (TF-IDF + cosine similarity)

## 🧪 Try it Locally
```bash
pip install -r requirements.txt
streamlit run app.py

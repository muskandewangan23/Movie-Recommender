import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load data
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# Merge & preprocess
movie_data = pd.merge(movies, ratings, on='movieId')
movie_stats = movie_data.groupby('title').agg({'rating': ['mean', 'count']})
movie_stats.columns = ['avg_rating', 'rating_count']
movie_stats = movie_stats.reset_index()
movie_df = pd.merge(movies, movie_stats, on='title')

# TF-IDF on genres
movie_df['genres_clean'] = movie_df['genres'].str.replace('|', ' ', regex=False)
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movie_df['genres_clean'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Reverse index
indices = pd.Series(movie_df.index, index=movie_df['title']).drop_duplicates()

# Recommend function
def hybrid_recommend(movie_title, top_n=5):
    if movie_title not in indices:
        return ["Movie not found."]
    
    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:25]
    movie_indices = [i[0] for i in sim_scores]
    sim_movies = movie_df.iloc[movie_indices].copy()
    sim_movies['similarity'] = [i[1] for i in sim_scores]
    sim_movies['score'] = sim_movies['similarity'] * np.log1p(sim_movies['rating_count'])
    return sim_movies.sort_values('score', ascending=False)['title'].head(top_n).tolist()

# Streamlit UI
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

st.title("🎬 Movie Recommendation System")
st.markdown("Select a movie to get top 5 similar recommendations:")

selected_movie = st.selectbox("Choose a movie", sorted(movie_df['title'].unique()))

if st.button("Recommend"):
    with st.spinner("Finding movies for you..."):
        recommendations = hybrid_recommend(selected_movie)
        st.success("Here are your recommendations:")
        for i, title in enumerate(recommendations, 1):
            st.write(f"{i}. {title}")

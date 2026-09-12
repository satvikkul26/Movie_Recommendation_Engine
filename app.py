import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_pickle("Model/df.pkl")

tfidf_matrix = pickle.load(
    open("Model/tfidf_matrix.pkl", "rb")
)

indices = pickle.load(
    open("Model/indices.pkl", "rb")
)

def recommend(title, n=5):
    idx = indices[title]
    similarity_scores = cosine_similarity(
        tfidf_matrix[idx],
        tfidf_matrix
    ).flatten()
    similar_indices=similarity_scores.argsort()[::-1][1:n+1]
    recommendations=df.iloc[similar_indices].copy()
    recommendations["similarity"]=similarity_scores[similar_indices]
    return recommendations

st.set_page_config(
    page_title="ITS MY CHOICE",
    page_icon="🎬",
    layout="wide"
)

st.title("😎 ITS MY CHOICE")
st.subheader("You pick one. We find the rest.")
st.markdown("Movie Recommendation System Based On")
st.markdown("""
- **45453 Unique Movies**
- **24 Parameters**
""")
st.write("Find movies similar to your favourite movie.")


movie_list=sorted(
    df["title"].dropna().unique()
)

selected_movie=st.selectbox(
    "Enter a movie",
    movie_list,
    index=None,
    placeholder="Search for a movie..."
)
n = st.slider("Number of recommendations", 3, 10, 5)

if st.button("🔍Get Recommendations", type="primary"):
    if selected_movie is None:
        st.warning("Please select a movie.")
    else:
        recommendations = recommend(selected_movie,n)
        st.markdown("### 🍿Your Top 5 Recommendations")
        for _, movie in recommendations.iterrows():
            st.markdown("---")
            st.header(movie["title"])
            match=movie["similarity"] * 100
            st.write(f"🎯**Match: {match:.1f}%**")
            if pd.notna(movie["tagline"]):
                st.write(f"**Tagline:** {movie['tagline']}")
            if pd.notna(movie["overview"]):
                st.write(f"**Overview:** {movie['overview']}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("**🎭Genre**")
                st.write(movie["genres"])

            with col2:
                st.write("**🔥Popularity**")
                st.write(movie["popularity"])

            with col3:
                st.write("**📅Release Date**")
                st.write(movie["release_date"])
            st.divider()
    st.success("💙 Thank you for exploring my Movie Recommendation Engine!")


st.caption(
    "Made with 💛 By Satvik"
    )
st.divider()
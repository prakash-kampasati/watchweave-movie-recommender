import os
import pickle
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
MOVIE_DATA_PATH = BASE_DIR / "movie_dict.pkl"

st.set_page_config(page_title="WatchWeave | Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_data
def load_movies(path_string: str) -> pd.DataFrame:
    """Load the compact movie dictionary and return a consistent DataFrame."""
    with open(path_string, "rb") as file:
        movie_dict = pickle.load(file)

    movies_df = pd.DataFrame(movie_dict)
    required_columns = {"movie_id", "title", "tags"}
    missing = required_columns.difference(movies_df.columns)
    if missing:
        raise ValueError(f"movie_dict.pkl is missing required columns: {', '.join(sorted(missing))}")

    movies_df["title"] = movies_df["title"].fillna("").astype(str)
    movies_df["tags"] = movies_df["tags"].fillna("").astype(str)
    return movies_df


@st.cache_resource
def build_tag_vectors(tags: tuple[str, ...]):
    """Build sparse text vectors once; do not create/save a huge dense similarity matrix."""
    vectorizer = CountVectorizer(max_features=5000, stop_words="english")
    tag_vectors = vectorizer.fit_transform(tags)
    return tag_vectors


def get_tmdb_api_key() -> str:
    """Read the TMDB key from Streamlit secrets or an environment variable."""
    try:
        key = st.secrets.get("TMDB_API_KEY", "")
    except Exception:
        key = ""
    return key or os.getenv("TMDB_API_KEY", "")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_poster(movie_id: int, api_key: str):
    """Fetch a poster from TMDB. Return None if the key or poster is unavailable."""
    if not api_key:
        return None

    try:
        response = requests.get(
            f"https://api.themoviedb.org/3/movie/{movie_id}",
            params={"api_key": api_key, "language": "en-US"},
            timeout=10,
        )
        response.raise_for_status()
        poster_path = response.json().get("poster_path")
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except (requests.RequestException, ValueError):
        return None
    return None


def recommend(movie_title: str, movies_df: pd.DataFrame, tag_vectors, api_key: str, count: int = 5):
    matches = movies_df.index[movies_df["title"] == movie_title].tolist()
    if not matches:
        return []

    movie_index = matches[0]
    # Compare only the selected movie with all movies. This returns one row of scores,
    # rather than storing a 4,806 x 4,806 dense matrix (~176 MB).
    scores = cosine_similarity(tag_vectors[movie_index], tag_vectors).ravel()
    ranked_indices = scores.argsort()[::-1]
    ranked_indices = [i for i in ranked_indices if i != movie_index][:count]

    results = []
    for index in ranked_indices:
        row = movies_df.iloc[index]
        results.append(
            {
                "title": row["title"],
                "movie_id": int(row["movie_id"]),
                "poster": fetch_poster(int(row["movie_id"]), api_key),
                "score": float(scores[index]),
            }
        )
    return results


st.title("🎬 WatchWeave")
st.caption("Your watch history. Your next favorite.")
st.write("Choose a movie you enjoyed to discover five similar movies.")

try:
    movies = load_movies(str(MOVIE_DATA_PATH))
    if movies.empty:
        st.error("No movies were found in movie_dict.pkl.")
        st.stop()

    with st.spinner("Preparing movie descriptions..."):
        tag_vectors = build_tag_vectors(tuple(movies["tags"].tolist()))

    api_key = get_tmdb_api_key()
    if not api_key:
        st.info(
            "Recommendations work without a TMDB API key, but posters need one. "
            "Add TMDB_API_KEY to Streamlit Secrets to show movie posters."
        )

    selected_movie = st.selectbox(
        "Select a movie",
        options=movies["title"].tolist(),
        index=0,
    )

    if st.button("Recommend movies", type="primary"):
        with st.spinner("Finding movies you may like..."):
            recommendations = recommend(selected_movie, movies, tag_vectors, api_key)

        if not recommendations:
            st.warning("No recommendations were found for that movie.")
        else:
            columns = st.columns(5)
            for column, movie in zip(columns, recommendations):
                with column:
                    st.subheader(movie["title"])
                    if movie["poster"]:
                        st.image(movie["poster"], use_container_width=True)
                    else:
                        st.caption("Poster unavailable")
                    st.caption(f"Similarity: {movie['score']:.2f}")

except FileNotFoundError:
    st.error(
        "Could not find movie_dict.pkl. Make sure it is in the same folder as app.py "
        "and is included in your GitHub repository."
    )
except Exception as exc:
    st.error(f"WatchWeave could not start: {exc}")

import os
import pickle
import requests
import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="WatchWeave | Movie Discovery",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
:root {
    --bg: #080E1A;
    --surface: #121D2E;
    --surface-hover: #192B40;
    --line: rgba(165,190,215,.18);
    --text: #F3F6FB;
    --muted: #AAB8CB;
    --teal: #4DE0C1;
    --blue: #6CA8FF;
    --gold: #F3C969;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, rgba(77,224,193,.09), transparent 35%),
        radial-gradient(ellipse at 90% 15%, rgba(108,168,255,.10), transparent 35%),
        var(--bg);
    color: var(--text);
}

.block-container {
    max-width: 1380px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

header[data-testid="stHeader"] {
    background: rgba(8,14,26,.85);
}

#MainMenu, footer {
    visibility: hidden;
}

h1, h2, h3, p, label {
    color: var(--text);
}

.ww-brand {
    display: flex;
    align-items: center;
    gap: 13px;
    color: var(--text);
    font-size: 20px;
    font-weight: 900;
    letter-spacing: 2px;
    margin: 0 0 36px 0;
}

.ww-brand small {
    display: block;
    color: var(--muted);
    font-size: 9px;
    letter-spacing: 2px;
    font-weight: 600;
    margin-top: 5px;
}

.ww-mark {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 44px;
    height: 44px;
    border-radius: 14px;
    background: linear-gradient(135deg, var(--teal), var(--blue));
    color: #08111D;
    font-size: 25px;
    font-weight: 1000;
    box-shadow: 0 5px 25px rgba(77,224,193,.15);
}

.ww-hero {
    padding: 36px 0 32px 0;
}

.ww-eyebrow {
    color: var(--teal);
    text-transform: uppercase;
    letter-spacing: 3px;
    font-size: 11px;
    font-weight: 800;
    margin-bottom: 18px;
}

.ww-title {
    font-size: clamp(38px, 5vw, 66px);
    line-height: 1.08;
    font-weight: 900;
    letter-spacing: -2px;
    margin: 0 0 20px 0;
}

.ww-title span {
    background: linear-gradient(90deg, var(--teal), var(--blue));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.ww-subtitle {
    color: var(--muted);
    font-size: 17px;
    line-height: 1.8;
    max-width: 680px;
}

.ww-pills {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 24px;
}

.ww-pill {
    border: 1px solid var(--line);
    border-radius: 50px;
    background: rgba(255,255,255,.035);
    color: #DCE6F4;
    padding: 9px 14px;
    font-size: 12px;
}

.ww-section-title {
    font-size: 27px;
    font-weight: 850;
    letter-spacing: -.5px;
    margin: 12px 0 8px 0;
}

.ww-copy {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.8;
}

div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: var(--surface);
    color: var(--text);
    border: 1px solid var(--line);
    border-radius: 12px;
    min-height: 48px;
}

div[data-testid="stTextInput"] input:focus {
    border-color: var(--teal);
    box-shadow: 0 0 0 1px var(--teal);
}

div[data-testid="stTextInput"] label,
div[data-testid="stSelectbox"] label {
    color: #DCE6F4;
    font-weight: 700;
}

.stButton > button {
    border: 1px solid rgba(77,224,193,.45);
    background: linear-gradient(135deg, rgba(77,224,193,.13), rgba(108,168,255,.13));
    color: var(--text);
    border-radius: 12px;
    min-height: 45px;
    font-weight: 750;
    transition: all .2s ease;
}

.stButton > button:hover {
    border-color: var(--teal);
    color: var(--teal);
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(77,224,193,.09);
}

.ww-card {
    background: linear-gradient(145deg, var(--surface), rgba(18,29,46,.75));
    border: 1px solid var(--line);
    border-radius: 17px;
    padding: 14px;
    margin-top: 12px;
    margin-bottom: 12px;

    /* Equal-height movie information boxes */
    height: 168px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;

    transition: all .25s ease;
    overflow: hidden;
}

.ww-card:hover {
    background: var(--surface-hover);
    border-color: rgba(77,224,193,.55);
    transform: translateY(-4px);
    box-shadow: 0 12px 30px rgba(0,0,0,.2);
}

.ww-rank {
    display: inline-block;
    color: var(--teal);
    font-size: 10px;
    font-weight: 850;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 8px;
    flex-shrink: 0;
}

.ww-card-title {
    font-size: 15px;
    line-height: 1.5;
    font-weight: 800;
    color: var(--text);
    overflow-wrap: anywhere;

    /* Keep titles within a consistent area */
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
    min-height: 67px;
}

.ww-score-row {
    margin-top: auto;
    padding-top: 8px;
    color: var(--muted);
    font-size: 12px;
    flex-shrink: 0;
}

.ww-score {
    color: var(--gold);
    font-weight: 800;
}

.ww-placeholder {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 230px;
    border: 1px solid var(--line);
    border-radius: 15px;
    background: linear-gradient(145deg, #14253A, #0C1422);
    color: var(--muted);
    font-size: 44px;
}

div[data-testid="stImage"] img {
    border-radius: 13px;
    border: 1px solid var(--line);
    transition: transform .25s ease, filter .25s ease;
}

div[data-testid="stImage"] img:hover {
    transform: scale(1.025);
    filter: brightness(1.08);
}

.ww-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--line), transparent);
    margin: 38px 0;
}

.ww-about {
    padding: 24px;
    border-radius: 18px;
    border: 1px solid var(--line);
    background: rgba(18,29,46,.75);
    margin-top: 15px;
}

.ww-about-title {
    color: var(--text);
    font-size: 20px;
    font-weight: 850;
    margin-bottom: 10px;
}

.ww-about-copy {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.9;
}

.ww-about-grid {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 14px;
    margin-top: 22px;
}

.ww-about-item {
    padding: 17px;
    border-radius: 13px;
    border: 1px solid var(--line);
    background: rgba(255,255,255,.025);
}

.ww-about-item strong {
    display: block;
    color: var(--teal);
    margin-bottom: 8px;
    font-size: 14px;
}

.ww-about-item span {
    color: var(--muted);
    font-size: 12px;
    line-height: 1.7;
}

.ww-footer {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 20px;
    border-top: 1px solid var(--line);
    margin-top: 45px;
    padding: 26px 0 10px 0;
}

.ww-footer-brand {
    font-weight: 900;
    letter-spacing: 2px;
    font-size: 15px;
    color: var(--text);
}

.ww-footer-tagline {
    color: var(--muted);
    font-size: 12px;
    line-height: 1.8;
    margin-top: 8px;
}

.ww-footer-credit {
    color: #71839C;
    font-size: 11px;
    margin-top: 8px;
}

@media (max-width: 800px) {
    .ww-about-grid {
        grid-template-columns: 1fr;
    }

    .ww-title {
        letter-spacing: -1px;
    }
}

@media (max-width: 600px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .ww-brand {
        margin-bottom: 20px;
    }

    .ww-hero {
        padding-top: 20px;
    }

    .ww-footer {
        flex-direction: column;
    }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MOVIE DATA
# ============================================================

@st.cache_data
def load_movies():
    with open("movie_dict.pkl", "rb") as file:
        movie_data = pickle.load(file)

    movies = pd.DataFrame(movie_data)

    required_columns = {"movie_id", "title", "tags"}
    missing_columns = required_columns - set(movies.columns)

    if missing_columns:
        raise ValueError(
            f"Your movie_dict.pkl is missing columns: {missing_columns}"
        )

    movies = movies.dropna(subset=["title", "tags"]).copy()
    movies["title"] = movies["title"].astype(str)
    movies["tags"] = movies["tags"].astype(str)
    movies["movie_id"] = pd.to_numeric(
        movies["movie_id"], errors="coerce"
    )

    movies = movies.dropna(subset=["movie_id"])
    movies["movie_id"] = movies["movie_id"].astype(int)
    movies = movies.reset_index(drop=True)

    return movies


# ============================================================
# BUILD VECTORS AND SIMILARITY ON DEMAND
# ============================================================

@st.cache_resource
def build_movie_vectors(tags):
    vectorizer = CountVectorizer(
        max_features=5000,
        stop_words="english"
    )

    movie_vectors = vectorizer.fit_transform(tags)

    return movie_vectors


# ============================================================
# TMDB POSTERS
# ============================================================

def get_tmdb_api_key():
    # Set TMDB_API_KEY in Streamlit secrets or environment variables.
    try:
        return st.secrets["TMDB_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.getenv("TMDB_API_KEY", "")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_poster(movie_id, api_key):
    if not api_key:
        return None

    url = f"https://api.themoviedb.org/3/movie/{movie_id}"

    try:
        response = requests.get(
            url,
            params={"api_key": api_key},
            timeout=8
        )
        response.raise_for_status()

        data = response.json()
        poster_path = data.get("poster_path")

        if poster_path:
            return f"https://image.tmdb.org/t/p/w500{poster_path}"

    except (requests.RequestException, ValueError):
        return None

    return None


# ============================================================
# RECOMMEND MOVIES
# ============================================================

def recommend_movies(movie_title, movies, movie_vectors, number=5):
    matches = movies.index[
        movies["title"].str.casefold() == movie_title.casefold()
    ].tolist()

    if not matches:
        return []

    selected_index = matches[0]

    # Calculate similarity only for the selected movie.
    similarity_scores = cosine_similarity(
        movie_vectors[selected_index],
        movie_vectors
    ).flatten()

    ranked_indices = similarity_scores.argsort()[::-1]

    recommendations = []
    for index in ranked_indices:
        if index == selected_index:
            continue

        score = float(similarity_scores[index])

        recommendations.append({
            "movie": movies.iloc[index],
            "score": score
        })

        if len(recommendations) == number:
            break

    return recommendations


# ============================================================
# BRAND HEADER
# ============================================================

st.markdown("""
<div class="ww-brand">
    <span class="ww-mark">W</span>
    <span>
        WATCHWEAVE
        <small>DISCOVER YOUR NEXT FAVORITE</small>
    </span>
</div>
""", unsafe_allow_html=True)


# ============================================================
# HERO SECTION
# ============================================================

st.markdown("""
<section class="ww-hero">
    <p class="ww-eyebrow">A better way to find what to watch</p>
    <h1 class="ww-title">
        Your next movie night<br>
        starts <span>here.</span>
    </h1>
    <p class="ww-subtitle">
        Discover movies that match your taste.
        Less scrolling, more discovering.
    </p>
    <div class="ww-pills">
        <span class="ww-pill">✦ Tailored movie picks</span>
        <span class="ww-pill">◈ Smart discovery</span>
        <span class="ww-pill">▶ Movie recommendations</span>
    </div>
</section>
""", unsafe_allow_html=True)


# ============================================================
# LOAD DATA
# ============================================================

try:
    movies = load_movies()
    movie_vectors = build_movie_vectors(
        tuple(movies["tags"].tolist())
    )
except FileNotFoundError:
    st.error(
        "Could not find movie_dict.pkl. "
        "Place it in the same folder as app.py."
    )
    st.stop()
except Exception as error:
    st.error(f"Unable to load movie data: {error}")
    st.stop()


# ============================================================
# MOVIE SEARCH
# ============================================================

st.markdown(
    '<h2 class="ww-section-title">Find your next favorite</h2>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="ww-copy">Search for a movie, select it, '
    'and discover films with similar content.</p>',
    unsafe_allow_html=True
)

search_text = st.text_input(
    "Search movies",
    placeholder="🔎  Start typing a movie title...",
    key="movie_search"
)

if search_text.strip():
    filtered_movies = movies[
        movies["title"].str.contains(
            search_text.strip(),
            case=False,
            na=False,
            regex=False
        )
    ]

    # Put exact title matches first.
    filtered_movies = filtered_movies.copy()
    filtered_movies["_exact_match"] = (
        filtered_movies["title"].str.casefold()
        == search_text.strip().casefold()
    )

    filtered_movies = filtered_movies.sort_values(
        "_exact_match",
        ascending=False
    )

    filtered_titles = filtered_movies["title"].tolist()
else:
    filtered_titles = movies["title"].tolist()


if not filtered_titles:
    st.warning(
        "No movies match that search. Try another title."
    )
    selected_movie = None
else:
    selected_movie = st.selectbox(
        "Choose a movie",
        options=filtered_titles,
        index=0,
        key="selected_movie"
    )


# ============================================================
# RECOMMENDATION BUTTON
# ============================================================

if selected_movie:
    if st.button(
        "✨  Discover Similar Movies",
        use_container_width=True
    ):
        st.session_state["recommended_movie"] = selected_movie

recommended_movie = st.session_state.get("recommended_movie")


# ============================================================
# RECOMMENDATION RESULTS
# ============================================================

if recommended_movie:
    st.markdown('<div class="ww-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        f'<h2 class="ww-section-title">'
        f'Because you selected <span style="color:#4DE0C1;">'
        f'{recommended_movie}</span></h2>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="ww-copy">Here are five movies that share '
        'similar content characteristics.</p>',
        unsafe_allow_html=True
    )

    recommendations = recommend_movies(
        recommended_movie,
        movies,
        movie_vectors,
        number=5
    )

    if not recommendations:
        st.warning("No recommendations could be generated.")

    else:
        api_key = get_tmdb_api_key()

        columns = st.columns(5, gap="medium")

        for rank, (column, item) in enumerate(
                zip(columns, recommendations),
                start=1
        ):
            movie = item["movie"]
            similarity_score = item["score"]

            with column:
                # Display movie title and similarity score
                st.markdown(
                    f"""
                    <div class="ww-card">
                        <div class="ww-rank">PICK #{rank}</div>
                        <div class="ww-card-title">
                            {movie["title"]}
                        </div>
                        <div class="ww-score-row">
                            Content match:
                            <span class="ww-score">
                                {similarity_score * 100:.1f}%
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # Fetch and display the movie poster
                poster_url = fetch_poster(
                    int(movie["movie_id"]),
                    api_key
                )

                if poster_url:
                    tmdb_url = (
                        f"https://www.themoviedb.org/movie/"
                        f"{int(movie['movie_id'])}"
                    )

                    st.markdown(
                        f"""
                        <a href="{tmdb_url}" target="_blank" rel="noopener noreferrer">
                            <img
                                src="{poster_url}"
                                alt="View {movie['title']} on TMDB"
                                style="
                                    width: 100%;
                                    height: auto;
                                    border-radius: 13px;
                                    border: 1px solid rgba(165,190,215,.18);
                                    cursor: pointer;
                                    transition: transform 0.25s ease;
                                "
                            >
                        </a>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        '<div class="ww-placeholder">🎬</div>',
                        unsafe_allow_html=True
                    )

        if not api_key:
            st.info(
                "To display movie posters, configure your TMDB API key."
            )


# ============================================================
# ABOUT WATCHWEAVE
# ============================================================

st.markdown('<div class="ww-divider"></div>', unsafe_allow_html=True)

st.markdown(
    '<h2 class="ww-section-title">About WatchWeave</h2>',
    unsafe_allow_html=True
)

with st.container(border=True):
    st.subheader("Every movie connects to another story.")

    st.write(
        "WatchWeave is a content-based movie recommendation system "
        "designed to help you discover films related to your interests. "
        "It analyzes movie tags and compares their content to identify "
        "titles that may appeal to you."
    )

    st.write(
        "The recommendations are based on text similarity, not on "
        "personal viewing history or user ratings."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 01 · Movie Features")
        st.write(
            "Movie tags are converted into numerical features "
            "using CountVectorizer."
        )

    with col2:
        st.markdown("### 02 · Similarity")
        st.write(
            "Cosine similarity measures how closely movie "
            "feature vectors match."
        )

    with col3:
        st.markdown("### 03 · Discovery")
        st.write(
            "The system ranks similar movies and displays "
            "the top recommendations."
        )

# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="ww-footer">
    <div>
        <div class="ww-footer-brand">◈ WATCHWEAVE</div>
        <div class="ww-footer-tagline">
            Your watchlist is waiting.<br>
            Find the thread to your next favorite.
        </div>
    </div>
    <div>
        <div class="ww-footer-tagline">
            Made for curious movie lovers
        </div>
        <div class="ww-footer-credit">
            Content-based movie recommendations · Built with Python
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
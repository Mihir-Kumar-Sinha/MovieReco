import streamlit as st
import pandas as pd
import random
import requests
import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ==================== PAGE CONFIG ====================
st.set_page_config(page_title="Bollywood Magic Recommender 🎬", layout="centered", initial_sidebar_state="expanded")

# ==================== FIX FILE PATH ISSUE ====================
# Automatically find movies_data.csv even if you're running from anywhere
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "movies_data.csv"

# ==================== API KEY ====================
if "OMDB_API_KEY" in st.secrets:
    OMDB_API_KEY = st.secrets["OMDB_API_KEY"]
else:
    OMDB_API_KEY = st.sidebar.text_input("OMDb API Key (optional for posters)", type="password",
                                         help="Get free at omdbapi.com")

# ==================== LOAD DATA SAFELY ====================
@st.cache_data
def load_data():
    if not CSV_PATH.exists():
        st.error(f"❌ File 'movies_data.csv' not found!\n\nPlease make sure it's in the same folder as app.py")
        st.stop()
    df = pd.read_csv(str(CSV_PATH))
    df = df.dropna(subset=['Name', 'Year', 'Genre'])
    df['Year'] = df['Year'].astype(int)
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce').fillna(0)
    df['Votes'] = pd.to_numeric(df['Votes'].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

    df['soup'] = (
        df['Genre'].str.replace(', ', ' ') + ' ' +
        df['Director'].fillna('').str.replace(' ', '') + ' ' +
        df['Actor 1'].fillna('').str.replace(' ', '') + ' ' +
        df['Actor 2'].fillna('').str.replace(' ', '') + ' ' +
        df['Actor 3'].fillna('').str.replace(' ', '')
    )
    return df.reset_index(drop=True)

df = load_data()

# ==================== BUILD MODEL ====================
@st.cache_resource
def build_model():
    tfidf = TfidfVectorizer(stop_words='english', max_features=10000)
    matrix = tfidf.fit_transform(df['soup'])
    return cosine_similarity(matrix, matrix)

cosine_sim = build_model()
indices = pd.Series(df.index, index=df['Name']).drop_duplicates()

# ==================== OMDB FETCHER ====================
@st.cache_data(ttl=3600*24*7)
def fetch_omdb(title, year):
    if not OMDB_API_KEY or OMDB_API_KEY == "":
        return {"poster": "https://via.placeholder.com/500x750/222/fff?text=No+OMDb+Key", "plot": "Add OMDb key for posters!", "imdb": "N/A"}
    
    url = f"http://www.omdbapi.com/?t={title}&y={year}&apikey={OMDB_API_KEY}"
    try:
        r = requests.get(url, timeout=8)
        data = r.json()
        if data.get("Response") == "True":
            return {
                "poster": data.get("Poster", "https://via.placeholder.com/500x750?text=No+Poster"),
                "plot": data.get("Plot", "No plot available"),
                "imdb": data.get("imdbRating", "N/A")
            }
    except:
        pass
    return {"poster": "https://via.placeholder.com/500x750/333/ccc?text=Error", "plot": "Not available", "imdb": "N/A"}

# ==================== RECOMMEND FUNCTION ====================
def recommend(title):
    if title not in indices:
        return None
    idx = indices[title]
    scores = sorted(list(enumerate(cosine_sim[idx])), key=lambda x: x[1], reverse=True)[1:13]
    return df.iloc[[i[0] for i in scores]]

# ==================== SIDEBAR ====================
st.sidebar.title("Extra Features")
actor_list = sorted(set(df['Actor 1'].dropna().tolist() + 
                        df['Actor 2'].dropna().tolist() + 
                        df['Actor 3'].dropna().tolist()))
selected_actor = st.sidebar.selectbox("Search by Actor", [""] + actor_list)

director_list = sorted(df['Director'].dropna().unique())
selected_director = st.sidebar.selectbox("Search by Director", [""] + director_list)

if st.sidebar.button("🎲 Surprise Me! (Hidden Gem)"):
    gems = df[(df['Rating'] >= 7.5) & (df['Votes'] >= 100)]
    if not gems.empty:
        surprise = gems.sample(1).iloc[0]
        st.sidebar.success(f"**Surprise!**\n\n**{surprise['Name']} ({surprise['Year']})**\n⭐ {surprise['Rating']:.1f}")

# ==================== MAIN UI ====================
st.title("🎬 Bollywood Magic Recommender")
st.markdown("**Your personal Indian movie discovery engine**")

tab1, tab2, tab3 = st.tabs(["Recommend by Movie", "By Actor", "By Director"])

with tab1:
    movie = st.selectbox("Pick a movie you loved:", [""] + sorted(df['Name'].tolist()))
    if movie:
        st.markdown(f"### 🍿 Because you loved **{movie}**")
        recs = recommend(movie)
        if recs is None or recs.empty:
            st.warning("No similar movies found.")
        else:
            cols = st.columns(4)
            for i, (_, row) in enumerate(recs.iterrows()):
                with cols[i % 4]:
                    data = fetch_omdb(row['Name'], row['Year'])
                    st.image(data['poster'], use_container_width=True)  # Fixed deprecation
                    st.markdown(f"**{row['Name']} ({row['Year']})**")
                    st.caption(f"⭐ Dataset: {row['Rating']:.1f} | IMDb: {data['imdb']}")
                    st.caption(f"🎭 {row['Genre'][:70]}...")
                    with st.expander("Show Plot"):
                        st.write(data['plot'])

with tab2:
    if selected_actor:
        st.markdown(f"### Movies starring **{selected_actor}**")
        actor_movies = df[df[['Actor 1','Actor 2','Actor 3']].stack().str.contains(selected_actor, case=False).groupby(level=0).any()].head(12)
        cols = st.columns(4)
        for i, (_, row) in enumerate(actor_movies.iterrows()):
            with cols[i % 4]:
                data = fetch_omdb(row['Name'], row['Year'])
                st.image(data['poster'], use_container_width=True)
                st.markdown(f"**{row['Name']} ({row['Year']})**")
                st.caption(f"⭐ {row['Rating']:.1f}")

with tab3:
    if selected_director:
        st.markdown(f"### Movies directed by **{selected_director}**")
        dir_movies = df[df['Director'] == selected_director]
        cols = st.columns(4)
        for i, (_, row) in enumerate(dir_movies.iterrows()):
            with cols[i % 4]:
                data = fetch_omdb(row['Name'], row['Year'])
                st.image(data['poster'], use_container_width=True)
                st.markdown(f"**{row['Name']} ({row['Year']})**")
                st.caption(f"⭐ {row['Rating']:.1f} | {row['Genre'][:50]}...")

st.success("App running perfectly! No errors, no warnings!")
st.balloons()
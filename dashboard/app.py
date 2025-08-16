# app.py
import re
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
import streamlit as st

from config import APP_TITLE, LAYOUT, BASE_CSS, PRIMARY, GOOD, BAD, GREY, BORDER
from utils import clamp
from data_loader import get_db, get_game_name, load_df
from analysis import get_vader, compute_sentiment, contains_any

# Tabs
from tabs import synthese, sentiment, themes, langues, playtime, longueur, cooccurrences, anomalies, qualite, explorateur, updates

st.set_page_config(page_title=APP_TITLE, layout=LAYOUT)
st.markdown(BASE_CSS, unsafe_allow_html=True)

# ----- DB & collections
db = get_db()
collections = [c for c in db.list_collection_names() if c.startswith("reviews_")]
app_ids = [c.replace("reviews_", "") for c in collections]
if not app_ids:
    st.error("Aucune collection 'reviews_<app_id>' dans MongoDB."); st.stop()

names = {app_id: get_game_name(app_id) for app_id in app_ids}
selected_app = st.selectbox("🎮 Jeu", options=sorted(app_ids), format_func=lambda a: names[a])

st.markdown(f"### {names[selected_app]}")
st.image(f"https://cdn.akamai.steamstatic.com/steam/apps/{selected_app}/header.jpg", use_container_width=True)

df = load_df(f"reviews_{selected_app}", db)
if df.empty:
    st.warning("Aucune donnée disponible pour ce jeu."); st.stop()

# ----- Date range defaults
if df["review_date"].notna().any():
    global_min = df["review_date"].min().date(); global_max = df["review_date"].max().date()
else:
    global_min = date(2024, 1, 1); global_max = datetime.now().date()

if st.session_state.get("current_app") != selected_app:
    st.session_state.current_app = selected_app
    st.session_state.date_min = global_min; st.session_state.date_max = global_max

st.session_state.date_min = clamp(st.session_state.get("date_min", global_min), global_min, global_max)
st.session_state.date_max = clamp(st.session_state.get("date_max", global_max), global_min, global_max)
if st.session_state.date_min > st.session_state.date_max:
    st.session_state.date_min, st.session_state.date_max = st.session_state.date_max, st.session_state.date_min

# ----- Global filters
with st.container():
    st.markdown("<div class='card'><h3>Filtres</h3>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns([1.3, 1, 1, 1, 1.2, 1.5])
    langs = sorted(df["language"].dropna().unique().tolist())
    with c1:
        chosen_langs = st.multiselect("🌐 Langues", options=langs, default=langs, key="global_langs")
    with c2:
        only_positive = st.checkbox("👍 Positifs", value=False, help="Filtre voted_up=True (si dispo).", key="global_pos_only")
    with c3:
        dmin = st.date_input("📅 Depuis", value=st.session_state.date_min, min_value=global_min, max_value=global_max, key="global_date_min")
    with c4:
        dmax = st.date_input("📅 Jusqu’à", value=st.session_state.date_max, min_value=global_min, max_value=global_max, key="global_date_max")
    with c5:
        st.caption("Période rapide"); b1, b2, b3 = st.columns(3)
        if b1.button("7 j", key="quick_7"):  st.session_state.date_min = max(global_min, global_max - timedelta(days=6));  st.session_state.date_max = global_max; st.rerun()
        if b2.button("30 j", key="quick_30"): st.session_state.date_min = max(global_min, global_max - timedelta(days=29)); st.session_state.date_max = global_max; st.rerun()
        if b3.button("90 j", key="quick_90"): st.session_state.date_min = max(global_min, global_max - timedelta(days=89)); st.session_state.date_max = global_max; st.rerun()
    with c6:
        keywords_raw = st.text_input("🔎 Mots‑clés (ex: bug, crash)", key="global_keywords")
        match_all   = st.checkbox("ET logique (tous)", value=False, key="global_keywords_all")

    dA, dB = st.columns([1,1])
    with dA: hard_refresh = st.checkbox("Purger le cache", value=False, help="Vide le cache avant recalcul.", key="global_hard_refresh")
    with dB:
        if st.button("🔄 Rafraîchir", key="global_refresh"):
            if hard_refresh: st.cache_data.clear()
            st.session_state.date_min = clamp(dmin, global_min, global_max)
            st.session_state.date_max = clamp(dmax, global_min, global_max)
            if st.session_state.date_min > st.session_state.date_max:
                st.session_state.date_min, st.session_state.date_max = st.session_state.date_max, st.session_state.date_min
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ----- Apply filters
mask = pd.Series(True, index=df.index)
if len(chosen_langs) > 0: mask &= df["language"].isin(chosen_langs)
if "voted_up" in df.columns and st.session_state.global_pos_only: mask &= (df["voted_up"] == True)
if df["review_date"].notna().any():
    mask &= (df["review_date"].dt.date >= st.session_state.date_min) & (df["review_date"].dt.date <= st.session_state.date_max)
df_f = df[mask].copy()

if keywords_raw and keywords_raw.strip() and not df_f.empty:
    kws = [k.strip().lower() for k in keywords_raw.split(",") if k.strip()]
    if kws:
        if match_all:
            lookaheads = "".join([rf"(?=.*\\b{re.escape(k)}\\b)" for k in kws]); pattern = lookaheads + r".*"
        else:
            pattern = r"\\b(" + "|".join([re.escape(k) for k in kws]) + r")\\b"
        df_f = df_f[df_f["cleaned_review"].str.contains(pattern, regex=True, na=False)]

st.caption(f"🗓️ Période : **{st.session_state.date_min} → {st.session_state.date_max}** • Avis filtrés : **{len(df_f):,}**")

# ----- Sentiment
sia = get_vader()
if not df_f.empty: df_f["sentiment"] = df_f["cleaned_review"].apply(lambda t: compute_sentiment(sia, t))
else: df_f["sentiment"] = []

pos = (df_f["sentiment"] > 0.05).mean()*100 if len(df_f) else 0.0
neu = ((df_f["sentiment"] >= -0.05) & (df_f["sentiment"] <= 0.05)).mean()*100 if len(df_f) else 0.0
neg = (df_f["sentiment"] < -0.05).mean()*100 if len(df_f) else 0.0
avg_len = df_f["cleaned_review"].str.split().apply(len).replace(0, np.nan).mean() if len(df_f) else 0.0


theme_dict = {
    "performances":["lag","fps","performance","stuttering","freeze","latence"],
    "gameplay":["gameplay","controls","mécaniques","mechanics","control"],
    "graphismes":["graphics","graphismes","art","textures"],
    "multijoueur":["multiplayer","coop","serveur","server"],
    "bugs":["bug","crash","error","issue","glitch"],
    "contenu":["content","dlc","missions","maps","map"]
}
rows = []
for th, kws in theme_dict.items():
    freq = df_f["cleaned_review"].apply(lambda x: contains_any(x, kws)).mean()*100 if len(df_f) else 0.0
    rows.append((th, freq))
freq_df = pd.DataFrame(rows, columns=["Thème","Fréquence (%)"]).sort_values("Fréquence (%)", ascending=False)

# ----- Tabs
tabs = st.tabs([
    "📌 Synthèse","🙂 Sentiment","🧩 Thèmes","🌍 Langues","⏱️ Heures de jeu",
    "✍️ Longueur & lisibilité","🔗 Cooccurrences","⚠️ Anomalies","✅ Qualité données","🔎 Explorateur","🛠️ Mises à jour"
])

ctx = {
    "df": df, "df_f": df_f,
    "pos": pos, "neu": neu, "neg": neg, "avg_len": avg_len,
    "theme_dict": theme_dict, "freq_df": freq_df
}

with tabs[0]: synthese.render(st, ctx)
with tabs[1]: sentiment.render(st, ctx)
with tabs[2]: themes.render(st, ctx)
with tabs[3]: langues.render(st, ctx)
with tabs[4]: playtime.render(st, ctx)
with tabs[5]: longueur.render(st, ctx)
with tabs[6]: cooccurrences.render(st, ctx)
with tabs[7]: anomalies.render(st, ctx)
with tabs[8]: qualite.render(st, ctx)
with tabs[9]: explorateur.render(st, ctx)
with tabs[10]: updates.render(st, ctx)

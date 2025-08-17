# app.py — Dashboard Steam Reviews (robuste, prêt pour reviews_clean) — PATCHED
import re
from datetime import datetime, date, timedelta
import numpy as np
import pandas as pd
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from config import APP_TITLE, LAYOUT, BASE_CSS
from utils import clamp
from data_loader import get_db, get_game_name, load_df
from analysis import get_vader, compute_sentiment, contains_any, clean_text_series

# Tabs
from tabs import synthese, sentiment, themes, langues, playtime, longueur, cooccurrences, anomalies, qualite, explorateur, updates

st.set_page_config(page_title=APP_TITLE, layout=LAYOUT)
st.markdown(BASE_CSS, unsafe_allow_html=True)

# ---------------------------------------------
# DB & discovery
# ---------------------------------------------
db = get_db()
all_cols = db.list_collection_names()

# 1) Mode per-app collections (reviews_<appid>) en excluant les collections globales
GLOBAL_CLEAN_NAMES = {"reviews_clean", "reviews_clean_airflow", "silver_clean", "clean_reviews"}
per_app_cols = [c for c in all_cols if c.startswith("reviews_") and c not in GLOBAL_CLEAN_NAMES]
per_app_ids = [c.split("reviews_", 1)[1] for c in per_app_cols]
per_app_ids_numeric = sorted([x for x in per_app_ids if x.isdigit()])

# 2) Mode collections "clean" (une seule table)
clean_like = [c for c in ["reviews_clean", "reviews_clean_airflow", "silver_clean", "clean_reviews"] if c in all_cols]

# Choix du mode
if per_app_ids_numeric:
    mode = "per_collection"
    app_ids = per_app_ids_numeric
elif clean_like:
    mode = "single_clean"
    base_clean = clean_like[0]
    try:
        ids = db[base_clean].distinct("app_id") or db[base_clean].distinct("appid") or db[base_clean].distinct("appId")
        def norm(x):
            if isinstance(x, int):
                return str(x) if x > 0 else None
            if isinstance(x, str) and x.isdigit():
                return x
            return None
        app_ids = sorted(set(filter(None, (norm(x) for x in ids))))
    except Exception:
        app_ids = []
else:
    mode = "none"
    app_ids = []

with st.expander("🛠️ Debug Mongo", expanded=False):
    st.write({
        "db": getattr(db, "name", "?"),
        "mode": mode,
        "collections": all_cols[:30],
        "per_app_ids_numeric": per_app_ids_numeric[:25],
        "clean_like": clean_like,
        "num_apps": len(app_ids),
    })

if not app_ids:
    st.error("Aucun jeu détecté.\n"
             "Attendus : des collections `reviews_<app_id>` (numériques) OU une collection 'reviews_clean' (ou variante) avec champ `app_id` numérique.")
    st.stop()

names = {app_id: get_game_name(app_id) for app_id in app_ids}
selected_app = st.selectbox("🎮 Jeu", options=app_ids, format_func=lambda a: names.get(a, a))

st.markdown(f"### {names.get(selected_app, selected_app)}")
st.image(f"https://cdn.akamai.steamstatic.com/steam/apps/{selected_app}/header.jpg", use_container_width=True)

# ---------------------------------------------
# Chargement des données
# ---------------------------------------------
def fetch_clean_reviews(db, app_id_str: str) -> pd.DataFrame:
    # On cible en priorité 'reviews_clean' si dispo ; sinon la première variante
    targets = [c for c in ["reviews_clean", "reviews_clean_airflow", "silver_clean", "clean_reviews"] if c in db.list_collection_names()]
    if not targets:
        return pd.DataFrame()
    col = targets[0]
    ors = [{"app_id": app_id_str}]
    if app_id_str.isdigit():
        ors.append({"app_id": int(app_id_str)})
    q = {"$or": ors}
    docs = list(db[col].find(q, {"_id": 0}))
    return pd.DataFrame(docs)

def _to_bool_series(s):
    try:
        if s.dtype == bool:
            return s
    except Exception:
        pass
    return s.astype(str).str.lower().map({"true": True, "1": True, "yes": True, "y": True, "false": False, "0": False, "no": False, "n": False}).fillna(s).astype("boolean")

def normalize_from_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise un DataFrame issu de reviews_clean (schema: cleaned_review, compound, review_date, language, voted_up, author.playtime_forever, ...)"""
    if df is None or df.empty:
        return pd.DataFrame(columns=["review_date","language","voted_up","cleaned_review","review_text","compound","playtime_hours"])
    df = df.copy()

    # Texte : on affiche le texte nettoyé produit par l’ETL
    if "cleaned_review" in df.columns:
        df["review_text"] = df["cleaned_review"].fillna("").astype(str)
    else:
        df["review_text"] = df.get("review_text", "").fillna("").astype(str)

    # Langue
    df["language"] = df.get("language", "unknown").astype(str)

    # voted_up
    if "voted_up" in df.columns:
        df["voted_up"] = _to_bool_series(df["voted_up"])
    else:
        df["voted_up"] = pd.NA

    # Dates
    if "review_date" in df.columns:
        df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce")
    elif "timestamp_created" in df.columns:
        df["review_date"] = pd.to_datetime(df["timestamp_created"], unit="s", errors="coerce")
    else:
        df["review_date"] = pd.NaT

    # Sentiment (utilise le score de l’ETL si présent)
    if "compound" in df.columns:
        df["compound"] = pd.to_numeric(df["compound"], errors="coerce").fillna(0.0)
    else:
        df["compound"] = 0.0

    # Playtime (minutes -> heures) depuis author.playtime_forever
    if "author" in df.columns:
        try:
            pt = df["author"].apply(lambda x: x.get("playtime_forever") if isinstance(x, dict) else None)
            df["playtime_hours"] = (pd.to_numeric(pt, errors="coerce").fillna(0)/60).round(2)
        except Exception:
            df["playtime_hours"] = np.nan
    elif "playtime_hours" not in df.columns:
        df["playtime_hours"] = np.nan

    return df

# Chargement selon le mode
if mode == "per_collection":
    df = load_df(f"reviews_{selected_app}", db)
else:
    df = fetch_clean_reviews(db, selected_app)
    df = normalize_from_clean(df)

if df.empty:
    st.warning("Aucune donnée disponible pour ce jeu."); st.stop()

# Assure la présence d'une colonne 'sentiment' AU NIVEAU GLOBAL (certains onglets l'utilisent directement)
if "compound" in df.columns:
    df["sentiment"] = pd.to_numeric(df["compound"], errors="coerce").fillna(0.0)
elif "sentiment" in df.columns:
    # si déjà présent (ex: 'pos'/'neg'), tente un mapping simple ; sinon calcule via VADER
    mapd = {"pos": 0.6, "neg": -0.6, "neu": 0.0}
    if df["sentiment"].dtype == object:
        df["sentiment"] = df["sentiment"].map(mapd).fillna(0.0)
else:
    try:
        sia = get_vader()
        df["sentiment"] = df["review_text"].apply(lambda t: compute_sentiment(sia, t))
    except Exception:
        df["sentiment"] = 0.0

# ---------------------------------------------
# Filtres & KPIs
# ---------------------------------------------
# Dates globales
if df["review_date"].notna().any():
    global_min = df["review_date"].min().date()
    global_max = df["review_date"].max().date()
else:
    global_min = date(2024, 1, 1)
    global_max = datetime.now().date()

if st.session_state.get("current_app") != selected_app:
    st.session_state.current_app = selected_app
    st.session_state.date_min = global_min
    st.session_state.date_max = global_max

st.session_state.date_min = clamp(st.session_state.get("date_min", global_min), global_min, global_max)
st.session_state.date_max = clamp(st.session_state.get("date_max", global_max), global_min, global_max)
if st.session_state.date_min > st.session_state.date_max:
    st.session_state.date_min, st.session_state.date_max = st.session_state.date_max, st.session_state.date_min

# Panneau filtres
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
        if b1.button("7 j", key="quick_7"):
            st.session_state.date_min = max(global_min, global_max - timedelta(days=6))
            st.session_state.date_max = global_max
            st.rerun()
        if b2.button("30 j", key="quick_30"):
            st.session_state.date_min = max(global_min, global_max - timedelta(days=29))
            st.session_state.date_max = global_max
            st.rerun()
        if b3.button("90 j", key="quick_90"):
            st.session_state.date_min = max(global_min, global_max - timedelta(days=89))
            st.session_state.date_max = global_max
            st.rerun()
    with c6:
        keywords_raw = st.text_input("🔎 Mots‑clés (ex: bug, crash)", key="global_keywords")
        match_all   = st.checkbox("ET logique (tous)", value=False, key="global_keywords_all")

    dA, dB = st.columns([1,1])
    with dA:
        hard_refresh = st.checkbox("Purger le cache", value=False, help="Vide le cache avant recalcul.", key="global_hard_refresh")
    with dB:
        if st.button("🔄 Rafraîchir", key="global_refresh"):
            if hard_refresh:
                st.cache_data.clear()
            st.session_state.date_min = clamp(dmin, global_min, global_max)
            st.session_state.date_max = clamp(dmax, global_min, global_max)
            if st.session_state.date_min > st.session_state.date_max:
                st.session_state.date_min, st.session_state.date_max = st.session_state.date_max, st.session_state.date_min
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# Application des filtres
mask = pd.Series(True, index=df.index)
if len(chosen_langs) > 0:
    mask &= df["language"].isin(chosen_langs)
if "voted_up" in df.columns and st.session_state.global_pos_only:
    mask &= (df["voted_up"] == True)
if df["review_date"].notna().any():
    mask &= (df["review_date"].dt.date >= st.session_state.date_min) & (df["review_date"].dt.date <= st.session_state.date_max)
df_f = df[mask].copy()

# Filtre par mots-clés
if keywords_raw and keywords_raw.strip() and not df_f.empty:
    kws = [k.strip().lower() for k in keywords_raw.split(",") if k.strip()]
    if kws:
        if match_all:
            lookaheads = "".join([rf"(?=.*\b{re.escape(k)}\b)" for k in kws])
            pattern = lookaheads + r".*"
        else:
            pattern = r"\b(" + "|".join([re.escape(k) for k in kws]) + r")\b"
        df_f = df_f[df_f["review_text"].str.contains(pattern, regex=True, na=False)]

st.caption(f"🗓️ Période : **{st.session_state.date_min} → {st.session_state.date_max}** • Avis filtrés : **{len(df_f):,}**")

# ---------------------------------------------
# Sentiment (utilisé par les autres onglets)
# ---------------------------------------------
# Utilise le score 'compound' de la BDD si présent, sinon VADER (déjà mis sur df)
if "sentiment" not in df_f.columns:
    if "compound" in df_f.columns:
        df_f["sentiment"] = pd.to_numeric(df_f["compound"], errors="coerce").fillna(0.0)
    else:
        sia = get_vader()
        df_f["sentiment"] = df_f["review_text"].apply(lambda t: compute_sentiment(sia, t))

# ---------------------------------------------
# Contexte partagé
# ---------------------------------------------
pos = (df_f["sentiment"] > 0.05).mean() * 100 if len(df_f) else 0.0
neu = ((df_f["sentiment"] >= -0.05) & (df_f["sentiment"] <= 0.05)).mean() * 100 if len(df_f) else 0.0
neg = (df_f["sentiment"] < -0.05).mean() * 100 if len(df_f) else 0.0
avg_len = df_f["review_text"].astype(str).str.split().apply(len).replace(0, np.nan).mean() if len(df_f) else 0.0

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
    freq = df_f["review_text"].apply(lambda x: contains_any(x, kws)).mean() * 100 if len(df_f) else 0.0
    rows.append((th, freq))
freq_df = pd.DataFrame(rows, columns=["Thème","Fréquence (%)"]).sort_values("Fréquence (%)", ascending=False)

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
with tabs[4]: playtime.render(st, ctx)          # nécessite playtime_hours -> fourni par normalize_from_clean
with tabs[5]: longueur.render(st, ctx)
with tabs[6]: cooccurrences.render(st, ctx)
with tabs[7]: anomalies.render(st, ctx)         # nécessite review_date + sentiment + volume
with tabs[8]: qualite.render(st, ctx)
with tabs[9]: explorateur.render(st, ctx)
with tabs[10]: updates.render(st, ctx)

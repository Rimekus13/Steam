# tabs/sentiment.py
import matplotlib.pyplot as plt
import seaborn as sns
from utils import DEFAULT_FIGSIZE_WIDE, title, compact_time_axis

def render(st, ctx):
    df_f = ctx["df_f"]
    st.markdown("<div class='card'><h3>Perception globale</h3>", unsafe_allow_html=True)
    s1,s2 = st.columns([1,1])
    with s1: sent_min = st.slider("Seuil min. sentiment", -1.0, 1.0, -1.0, 0.05, key="sent_min_slider")
    with s2: sent_max = st.slider("Seuil max. sentiment", -1.0, 1.0, 1.0, 0.05, key="sent_max_slider")
    df_sent = df_f[(df_f["sentiment"]>=sent_min) & (df_f["sentiment"]<=sent_max)]

    c1,c2,c3 = st.columns([1.4,1,1])
    with c1:
        if df_sent["review_date"].notna().any() and len(df_sent):
            ts=df_sent.dropna(subset=["review_date"]).copy()
            ts["date"]=ts["review_date"].dt.to_period("W").dt.start_time
            series=ts.groupby("date")["sentiment"].mean().rolling(3).mean()
            fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE_WIDE)
            ax.plot(series.index, series.values)
            compact_time_axis(ax,3,6); title(ax,"Tendance (MM x3)")
            ax.set_ylabel("Score"); ax.set_xlabel("")
            st.pyplot(fig, use_container_width=True)
            st.markdown("<div class='small'>Lissage léger pour mieux lire la tendance.</div>", unsafe_allow_html=True)
        else: st.info("Pas de dates exploitables.")
    with c2:
        dist_df = df_sent.assign(cat=df_sent["sentiment"].apply(lambda s: "Positif" if s>0.05 else ("Négatif" if s<-0.05 else "Neutre"))).groupby("cat").size().reindex(["Positif","Neutre","Négatif"]).fillna(0)
        fig, ax = plt.subplots(figsize=(4,2.2))
        ax.bar(dist_df.index, (dist_df.values/dist_df.values.sum()*100 if dist_df.values.sum() else dist_df.values))
        title(ax,"Répartition"); ax.set_ylabel("%"); ax.set_xlabel("")
        st.pyplot(fig, use_container_width=True)
        st.markdown("<div class='small'>Équilibre global des ressentis.</div>", unsafe_allow_html=True)
    with c3:
        if len(df_sent):
            fig, ax = plt.subplots(figsize=(4,2.2))
            sns.histplot(df_sent["sentiment"], bins=30, kde=True, ax=ax)
            title(ax,"Distribution des scores"); st.pyplot(fig, use_container_width=True)
            st.markdown("<div class='small'>Neutre ou polarisé ?</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

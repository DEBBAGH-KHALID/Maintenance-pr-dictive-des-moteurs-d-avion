import os
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Maintenance prédictive — Moteurs", page_icon="✈️", layout="wide")

# ============================================================
# Constantes
# ============================================================
NOMS = ["unit", "cycle", "set1", "set2", "set3"] + [f"s{i}" for i in range(1, 22)]
DEMO = "test_FD001.txt"

COULEURS = {"🔴 Critique": "#ef4444", "🟠 Surveillance": "#f59e0b", "🟢 Normal": "#22c55e"}
ORDRE = list(COULEURS.keys())
STYLE_CELLULE = {      # fond + texte définis ENSEMBLE : lisible en mode clair et sombre
    "🔴 Critique":     "background-color:#fecaca; color:#7f1d1d; font-weight:700",
    "🟠 Surveillance": "background-color:#fde68a; color:#78350f; font-weight:700",
    "🟢 Normal":       "background-color:#bbf7d0; color:#14532d; font-weight:700",
}
THEMES = {
    "Clair":  dict(bg="#f8fafc", bg2="#e2e8f0", txt="#0f172a", plotly="plotly_white"),
    "Sombre": dict(bg="#1e293b", bg2="#334155", txt="#f1f5f9", plotly="plotly_dark"),
}
# ============================================================
# Modèle
# ============================================================
@st.cache_resource
def charger_modele():
    return joblib.load("model_rul.pkl")

art = charger_modele()
model, colonnes = art["model"], art["colonnes"]


choix = st.sidebar.radio("🎨 Thème", ["Auto", "Clair", "Sombre"], horizontal=True)
if choix == "Auto":
    try:
        choix = "Sombre" if st.context.theme.type == "dark" else "Clair"
    except Exception:
        choix = "Clair"
T = THEMES[choix]

st.markdown(f"""
<style>
[data-testid="stAppViewContainer"], [data-testid="stHeader"] {{ background:{T['bg']}; }}
[data-testid="stSidebar"], [data-testid="stSidebar"] > div {{ background:{T['bg2']}; }}
.stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label, .stApp li,
.stApp [data-testid="stMetricValue"], .stApp [data-testid="stMetricLabel"],
.stApp [data-testid="stCaptionContainer"], .stApp [data-testid="stExpander"] summary,
[data-testid="stSidebar"] * {{ color:{T['txt']}; }}
[data-baseweb="select"] > div, [data-baseweb="input"] > div {{ background:{T['bg2']}; color:{T['txt']}; }}
[data-baseweb="popover"] ul {{ background:{T['bg2']}; color:{T['txt']}; }}
</style>
""", unsafe_allow_html=True)

def afficher(fig):
    fig.update_layout(template=T["plotly"], paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font_color=T["txt"])
    st.plotly_chart(fig, width="stretch", theme=None)
# ============================================================
# Barre latérale
# ============================================================
st.sidebar.header("⚙️ Réglages")

sources = ["Données de démonstration"] if os.path.exists(DEMO) else []
sources.append("Importer mon fichier")
source = st.sidebar.radio("Source des données", sources)

fichier = DEMO
if source == "Importer mon fichier":
    fichier = st.sidebar.file_uploader("Fichier .txt (format test_FD001)", type="txt")
    if fichier is None:
        st.title("✈️ Maintenance prédictive des moteurs")
        st.info("Importe un fichier dans la barre latérale pour commencer.")
        st.stop()

st.sidebar.subheader("Seuils d'alerte (en cycles)")
seuil_crit = st.sidebar.slider("🔴 Critique si RUL prédit <", 5, 60, 30, 5)
seuil_surv = st.sidebar.slider("🟠 Surveillance si RUL prédit <", seuil_crit + 5, 120, max(60, seuil_crit + 5), 5)
st.sidebar.caption("Au-dessus du seuil de surveillance : 🟢 Normal.")

# ============================================================
# Données + prédictions
# ============================================================
try:
    data = pd.read_csv(fichier, sep=r"\s+", header=None, names=NOMS)
    data["RUL_prédit"] = model.predict(data[colonnes])
except Exception as e:
    st.error(f"Fichier non valide : il doit avoir 26 colonnes comme test_FD001.txt. Détail : {e}")
    st.stop()

def statut(rul):
    if rul < seuil_crit:
        return "🔴 Critique"
    if rul < seuil_surv:
        return "🟠 Surveillance"
    return "🟢 Normal"

dernier = data.groupby("unit").tail(1).copy()
dernier["Statut"] = dernier["RUL_prédit"].apply(statut)
flotte = (dernier.rename(columns={"unit": "Moteur", "cycle": "Cycles écoulés", "RUL_prédit": "RUL prédit (cycles)"})
                 [["Moteur", "Cycles écoulés", "RUL prédit (cycles)", "Statut"]]
                 .sort_values("RUL prédit (cycles)").reset_index(drop=True))
flotte["RUL prédit (cycles)"] = flotte["RUL prédit (cycles)"].round(0).astype(int)

# ============================================================
# En-tête
# ============================================================
st.title("✈️ Maintenance prédictive des moteurs")
st.caption("Le modèle estime, pour chaque moteur, combien de **cycles** il lui reste avant la panne (RUL = *Remaining Useful Life*).")

with st.expander("ℹ️ Comment lire ce dashboard ?"):
    st.markdown(
        f"- **RUL prédit** : nombre de cycles de vol restants estimé par le modèle.\n"
        f"- 🔴 **Critique** : RUL < {seuil_crit} cycles → planifier la maintenance maintenant.\n"
        f"- 🟠 **Surveillance** : RUL entre {seuil_crit} et {seuil_surv} cycles → à suivre de près.\n"
        f"- 🟢 **Normal** : RUL ≥ {seuil_surv} cycles.\n"
        f"- Les seuils se règlent dans la barre latérale.\n"
        f"- Le modèle plafonne le RUL à 125 cycles (un moteur sain ne montre aucun signe de dégradation)."
    )

# ============================================================
# Indicateurs
# ============================================================
nb_crit = int((flotte["Statut"] == "🔴 Critique").sum())
nb_surv = int((flotte["Statut"] == "🟠 Surveillance").sum())
k1, k2, k3, k4 = st.columns(4)
k1.metric("Moteurs analysés", len(flotte))
k2.metric("🔴 Critiques", nb_crit)
k3.metric("🟠 En surveillance", nb_surv)
k4.metric("RUL moyen prédit", f"{flotte['RUL prédit (cycles)'].mean():.0f} cycles")

# ============================================================
# 1. Vue d'ensemble
# ============================================================
st.header("1. Vue d'ensemble de la flotte")
g1, g2 = st.columns([3, 2])

with g1:
    top = flotte.head(15).copy()
    top["Moteur"] = "Moteur " + top["Moteur"].astype(str)
    fig = px.bar(top, x="RUL prédit (cycles)", y="Moteur", orientation="h", color="Statut",
                 color_discrete_map=COULEURS, category_orders={"Statut": ORDRE},
                 text="RUL prédit (cycles)",
                 title="Les 15 moteurs les plus urgents (RUL prédit le plus bas)")
    fig.update_yaxes(autorange="reversed", title="")
    fig.update_xaxes(title="RUL prédit (cycles restants)")
    fig.add_vline(x=seuil_crit, line_dash="dash", line_color="#ef4444",
                  annotation_text="seuil critique", annotation_position="top")
    fig.add_vline(x=seuil_surv, line_dash="dash", line_color="#f59e0b",
                  annotation_text="seuil surveillance", annotation_position="top")
    fig.update_traces(textposition="outside")
    fig.update_layout(legend_title_text="Statut", height=470, margin=dict(t=70))
    afficher(fig)

with g2:
    repart = flotte["Statut"].value_counts().reindex(ORDRE, fill_value=0).reset_index()
    repart.columns = ["Statut", "Nombre de moteurs"]
    fig = px.pie(repart, names="Statut", values="Nombre de moteurs", hole=0.55,
                 color="Statut", color_discrete_map=COULEURS, category_orders={"Statut": ORDRE},
                 title="Répartition de la flotte par statut")
    fig.update_traces(textinfo="value+percent", sort=False)
    fig.update_layout(legend_title_text="Statut", height=470, margin=dict(t=70))
    afficher(fig)

# ============================================================
# 2. Tableau
# ============================================================
st.header("2. Classement des moteurs (du plus urgent au moins urgent)")
filtre = st.multiselect("Filtrer par statut", ORDRE, default=ORDRE)
tableau = flotte[flotte["Statut"].isin(filtre)]
styler = tableau.style.map(lambda v: STYLE_CELLULE.get(v, ""), subset=["Statut"])
if choix == "Sombre":
    styler = styler.set_properties(
        subset=["Moteur", "Cycles écoulés", "RUL prédit (cycles)"],
        **{"background-color": T["bg2"], "color": T["txt"]})
st.dataframe(styler, hide_index=True, width="stretch", height=380)
st.download_button("⬇️ Télécharger le classement (CSV)", flotte.to_csv(index=False),
                   "classement_moteurs.csv", "text/csv")

# ============================================================
# 3. Détail d'un moteur
# ============================================================
st.header("3. Analyse d'un moteur")
c1, c2 = st.columns([1, 3])
with c1:
    moteur = st.selectbox("Moteur", flotte["Moteur"].tolist(),
                          help="Classés du plus urgent au moins urgent.")
d = data[data["unit"] == moteur].reset_index(drop=True)
cycle_max = int(d["cycle"].max())
with c2:
    cycle = st.slider("Cycle actuel (fais glisser pour simuler l'évolution dans le temps)",
                      1, cycle_max, cycle_max)

ligne = d[d["cycle"] == cycle].iloc[0]
rul_now = float(ligne["RUL_prédit"])
st_now = statut(rul_now)

m1, m2 = st.columns([1, 2])
with m1:
    jauge = go.Figure(go.Indicator(
        mode="gauge+number", value=round(rul_now),
        number={"suffix": " cycles"},
        title={"text": f"RUL prédit — Moteur {moteur}, cycle {cycle}"},
        gauge={"axis": {"range": [0, 130]},
               "bar": {"color": COULEURS[st_now]},
               "steps": [{"range": [0, seuil_crit], "color": "rgba(239,68,68,0.25)"},
                         {"range": [seuil_crit, seuil_surv], "color": "rgba(245,158,11,0.25)"},
                         {"range": [seuil_surv, 130], "color": "rgba(34,197,94,0.25)"}]}))
    jauge.update_layout(height=330, margin=dict(t=80, b=10))
    afficher(jauge)
    st.markdown(f"**Statut : {st_now}**")

with m2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["cycle"], y=d["RUL_prédit"], mode="lines", name="RUL prédit",
                             line=dict(width=3), hovertemplate="Cycle %{x}<br>RUL prédit : %{y:.0f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[cycle], y=[rul_now], mode="markers", name="Cycle actuel",
                             marker=dict(size=13, color=COULEURS[st_now], line=dict(width=2, color="white"))))
    fig.add_hline(y=seuil_crit, line_dash="dash", line_color="#ef4444",
                  annotation_text="seuil critique", annotation_position="top left")
    fig.add_hline(y=seuil_surv, line_dash="dash", line_color="#f59e0b",
                  annotation_text="seuil surveillance", annotation_position="top left")
    fig.update_layout(title=f"Évolution du RUL prédit — Moteur {moteur}",
                      xaxis_title="Cycle (temps de fonctionnement)",
                      yaxis_title="RUL prédit (cycles restants)",
                      height=330, margin=dict(t=60, b=10), legend=dict(orientation="h", y=-0.25))
    afficher(fig)

st.subheader("Capteurs")
capteur = st.selectbox("Capteur à afficher", colonnes,
                       index=colonnes.index("s11") if "s11" in colonnes else 0)
fig = px.line(d, x="cycle", y=capteur, title=f"Capteur {capteur} — Moteur {moteur}",
              labels={"cycle": "Cycle (temps de fonctionnement)", capteur: f"Valeur du capteur {capteur}"})
fig.add_vline(x=cycle, line_dash="dot", annotation_text="cycle actuel", annotation_position="top")
fig.update_layout(height=320, margin=dict(t=60, b=10))
afficher(fig)
st.caption("Un capteur utile évolue nettement quand le moteur se dégrade.")
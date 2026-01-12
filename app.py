import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")
st.title("🚗 Car Price Prediction")


# =========================
# 1) Upload
# =========================
st.subheader("Chargement du dataset")

use_demo = st.checkbox("Utiliser le dataset d'exemple (Cardekho)")

uploaded_file = st.file_uploader("Ou uploader un autre fichier CSV", type=["csv"])

if use_demo:
    df = pd.read_csv("cardekho.csv")
    st.success("Dataset d'exemple chargé ✅")
elif uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Fichier CSV chargé avec succès ✅")
else:
    st.info("Veuillez choisir un mode de chargement pour continuer.")
    st.stop()

st.success("Fichier chargé avec succès ✅")

st.subheader("Aperçu du dataset")
st.write(f"Dimensions : {df.shape[0]} lignes × {df.shape[1]} colonnes")
st.dataframe(df.head(20), use_container_width=True)


# =========================
# 2) Infos colonnes + NaN
# =========================
st.subheader("Informations sur les colonnes (types & NaN)")

info_df = pd.DataFrame({
    "Colonne": df.columns,
    "Type": [str(df[c].dtype) for c in df.columns],
    "NaN (count)": [int(df[c].isna().sum()) for c in df.columns],
    "NaN (%)": [(df[c].isna().mean() * 100) for c in df.columns],
    "Valeurs uniques": [int(df[c].nunique(dropna=True)) for c in df.columns],
})
info_df["NaN (%)"] = info_df["NaN (%)"].round(2)

st.dataframe(info_df, use_container_width=True)


# =========================
# 3) Traitement NaN
# =========================
st.subheader("Traitement des valeurs manquantes (NaN)")

nan_cols = [c for c in df.columns if df[c].isna().sum() > 0]
st.write("Colonnes contenant des NaN :", nan_cols if nan_cols else "Aucune ✅")

strategy = st.selectbox("Stratégie de remplacement (numérique)", ["median", "mean"])

if st.button("Appliquer le traitement des NaN"):
    df_clean = df.copy()

    # convertir object -> numeric si possible (ex: max_power)
    for c in nan_cols:
        if df_clean[c].dtype == "object":
            df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce")

    # remplacer NaN sur colonnes numériques seulement
    for c in nan_cols:
        if pd.api.types.is_numeric_dtype(df_clean[c]):
            if strategy == "median":
                df_clean[c] = df_clean[c].fillna(df_clean[c].median())
            else:
                df_clean[c] = df_clean[c].fillna(df_clean[c].mean())

    st.success("Traitement appliqué ✅")
    st.write("NaN après traitement :")
    st.dataframe(df_clean.isna().sum(), use_container_width=True)

    st.session_state["df_clean"] = df_clean


# Dataset de travail (si NaN appliqué, sinon df)
df_work = st.session_state.get("df_clean", df)


# =========================
# 4) Suppression colonnes inutiles
# =========================
st.subheader("Suppression des colonnes inutiles")

cols_to_drop = st.multiselect(
    "Choisir les colonnes à supprimer",
    options=list(df_work.columns),
    default=["name"] if "name" in df_work.columns else []
)

if st.button("Supprimer les colonnes sélectionnées"):
    df_work2 = df_work.drop(columns=cols_to_drop, errors="ignore")
    st.success("Colonnes supprimées ✅")
    st.write("Colonnes restantes :")
    st.write(list(df_work2.columns))
    st.session_state["df_work"] = df_work2

df_work = st.session_state.get("df_work", df_work)


# =========================
# 5) Encodage One-Hot
# =========================
st.subheader("Encodage des variables catégorielles (One-Hot Encoding)")

if st.button("Appliquer l'encodage"):
    df_encoded = pd.get_dummies(df_work, drop_first=True)

    st.success("Encodage terminé ✅")
    st.write("Dimensions après encodage :", df_encoded.shape)
    st.dataframe(df_encoded.head(10), use_container_width=True)

    st.session_state["df_encoded"] = df_encoded


# =========================
# 6) Choix Target
# =========================
st.subheader("Choix de la variable cible (Target)")

df_encoded = st.session_state.get("df_encoded")

if df_encoded is None:
    st.info("Veuillez appliquer l'encodage avant de choisir la target.")
    st.stop()

target = st.selectbox(
    "Choisir la variable cible (Target)",
    options=df_encoded.columns,
    index=list(df_encoded.columns).index("selling_price") if "selling_price" in df_encoded.columns else 0
)

X = df_encoded.drop(columns=[target])
y = df_encoded[target]
# ---- Sécurité : supprimer NaN / inf dans X et y (obligatoire pour sklearn)
X = X.replace([np.inf, -np.inf], np.nan)

# si NaN existent encore, on remplit par la médiane des colonnes
if X.isna().any().any():
    X = X.fillna(X.median(numeric_only=True))

# pour être sûr que y ne contient pas NaN
y = y.replace([np.inf, -np.inf], np.nan)
if y.isna().any():
    y = y.fillna(y.median())

st.success(f"Target sélectionnée : {target}")
st.write("Dimensions X :", X.shape)
st.write("Dimensions y :", y.shape)

st.session_state["X"] = X
st.session_state["y"] = y


# =========================
# 7) Train/Test + Normalisation
# =========================
st.subheader("Division du dataset (Train / Test) et Normalisation")

test_size = st.slider("Taille du jeu de test (%)", 10, 40, 20)

if st.button("Appliquer Train/Test Split + Normalisation"):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.success("Division + Normalisation terminées ✅")
    st.write("X_train :", X_train_scaled.shape)
    st.write("X_test :", X_test_scaled.shape)

    st.session_state["X_train"] = X_train_scaled
    st.session_state["X_test"] = X_test_scaled
    st.session_state["y_train"] = y_train
    st.session_state["y_test"] = y_test
    st.session_state["scaler"] = scaler


# =========================
# 8) Choix modèle + Training
# =========================
st.subheader("Choix de l'algorithme et entraînement du modèle")

X_train = st.session_state.get("X_train")
y_train = st.session_state.get("y_train")
X_test = st.session_state.get("X_test")
y_test = st.session_state.get("y_test")

if X_train is None or y_train is None or X_test is None or y_test is None:
    st.info("Veuillez d'abord effectuer la division Train/Test (Étape 7).")
    st.stop()

mode = st.radio("Mode d'entraînement", ["Un seul modèle", "Comparer 2 modèles"], horizontal=True)

def eval_model(m, Xte, yte):
    pred = m.predict(Xte)
    mae = mean_absolute_error(yte, pred)
    rmse = np.sqrt(mean_squared_error(yte, pred))
    r2 = r2_score(yte, pred)
    return mae, rmse, r2

if mode == "Un seul modèle":
    model_choice = st.selectbox("Choisir un algorithme", ["Régression Linéaire", "Random Forest"])

    if st.button("Entraîner le modèle"):
        if model_choice == "Régression Linéaire":
            model = LinearRegression()
        else:
            model = RandomForestRegressor(n_estimators=200, random_state=42)

        model.fit(X_train, y_train)
        st.success("Modèle entraîné avec succès ✅")
        st.session_state["model"] = model
        st.session_state["model_name"] = model_choice

else:
    if st.button("Comparer et choisir le meilleur modèle"):
        results = []

        # 1) Linear Regression
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        mae, rmse, r2 = eval_model(lr, X_test, y_test)
        results.append(["Régression Linéaire", mae, rmse, r2, lr])

        # 2) Random Forest
        rf = RandomForestRegressor(n_estimators=300, random_state=42)
        rf.fit(X_train, y_train)
        mae, rmse, r2 = eval_model(rf, X_test, y_test)
        results.append(["Random Forest", mae, rmse, r2, rf])

        # Tableau résultats
        res_df = pd.DataFrame(results, columns=["Modèle", "MAE", "RMSE", "R²", "_model_obj"])
        st.dataframe(res_df.drop(columns=["_model_obj"]).style.format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R²": "{:.4f}"}), use_container_width=True)

        # Choisir le meilleur (R² max)
        best_row = res_df.sort_values("R²", ascending=False).iloc[0]
        best_name = best_row["Modèle"]
        best_model = best_row["_model_obj"]

        st.success(f"✅ Meilleur modèle sélectionné : **{best_name}**")
        st.session_state["model"] = best_model
        st.session_state["model_name"] = best_name


# =========================
# 9) Évaluation + Graphe
# =========================
st.subheader("Évaluation du modèle et visualisation")

model = st.session_state.get("model")
X_test = st.session_state.get("X_test")
y_test = st.session_state.get("y_test")

if model is None or X_test is None or y_test is None:
    st.info("Veuillez entraîner le modèle (Étape 8) et avoir le X_test/y_test (Étape 7).")
    st.stop()

if st.button("Évaluer le modèle"):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    st.write(f"**MAE :** {mae:.2f}")
    st.write(f"**RMSE :** {rmse:.2f}")
    st.write(f"**R² :** {r2:.4f}")

    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel("Valeurs réelles (y_test)")
    ax.set_ylabel("Valeurs prédites (y_pred)")
    ax.set_title("Réel vs Prédit")
    st.pyplot(fig)

    st.session_state["y_pred"] = y_pred


# =========================
# 10) Prédiction (formulaire propre)
# =========================
st.subheader("Prédiction avec de nouvelles données (formulaire)")

scaler = st.session_state.get("scaler")
df_work_for_form = st.session_state.get("df_work", df_work)

if scaler is None:
    st.info("Veuillez faire la normalisation (Étape 7) avant de prédire.")
    st.stop()

cat_cols = [c for c in ["fuel", "seller_type", "transmission", "owner"] if c in df_work_for_form.columns]

with st.form("predict_form"):
    st.write("Renseignez les caractéristiques du véhicule :")

    c1, c2 = st.columns(2)

    with c1:
        year_val = st.number_input("year", value=int(df_work_for_form["year"].median()) if "year" in df_work_for_form else 2015)
        km_val = st.number_input("km_driven", value=int(df_work_for_form["km_driven"].median()) if "km_driven" in df_work_for_form else 60000)
        mileage_val = st.number_input("mileage(km/ltr/kg)", value=float(df_work_for_form["mileage(km/ltr/kg)"].median()) if "mileage(km/ltr/kg)" in df_work_for_form else 19.0)
        engine_val = st.number_input("engine", value=float(df_work_for_form["engine"].median()) if "engine" in df_work_for_form else 1200.0)

    with c2:
        power_val = st.number_input("max_power", value=float(df_work_for_form["max_power"].median()) if "max_power" in df_work_for_form else 80.0)
        seats_val = st.number_input("seats", value=float(df_work_for_form["seats"].median()) if "seats" in df_work_for_form else 5.0)

        if "fuel" in cat_cols:
            fuel_val = st.selectbox("fuel", sorted(df_work_for_form["fuel"].dropna().unique().tolist()))
        else:
            fuel_val = None

        if "seller_type" in cat_cols:
            seller_val = st.selectbox("seller_type", sorted(df_work_for_form["seller_type"].dropna().unique().tolist()))
        else:
            seller_val = None

        if "transmission" in cat_cols:
            trans_val = st.selectbox("transmission", sorted(df_work_for_form["transmission"].dropna().unique().tolist()))
        else:
            trans_val = None

        if "owner" in cat_cols:
            owner_val = st.selectbox("owner", sorted(df_work_for_form["owner"].dropna().unique().tolist()))
        else:
            owner_val = None

    do_predict = st.form_submit_button("Prédire le prix")

if do_predict:
    row = {
        "year": year_val,
        "km_driven": km_val,
        "mileage(km/ltr/kg)": mileage_val,
        "engine": engine_val,
        "max_power": power_val,
        "seats": seats_val
    }
    if fuel_val is not None: row["fuel"] = fuel_val
    if seller_val is not None: row["seller_type"] = seller_val
    if trans_val is not None: row["transmission"] = trans_val
    if owner_val is not None: row["owner"] = owner_val

    input_raw = pd.DataFrame([row])

    # One-hot sur la ligne puis alignement avec X.columns
    input_encoded = pd.get_dummies(input_raw, drop_first=True)
    input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

    input_scaled = scaler.transform(input_encoded)
    pred_price = model.predict(input_scaled)[0]

    st.success(f"✅ Prix prédit : {pred_price:,.0f}")

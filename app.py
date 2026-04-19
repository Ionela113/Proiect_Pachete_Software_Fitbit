import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import statsmodels.api as sm

st.set_page_config(page_title="Proiect Fitbit", layout="wide")

st.title("Modelarea și evaluarea stilului de viață sănătos")
st.write("Aplicație Streamlit realizată pe baza datelor Fitbit")

# -----------------------------
# ÎNCĂRCARE DATE
# -----------------------------
@st.cache_data
def load_data():
    daily = pd.read_csv("data/dailyActivity_merged.csv")
    sleep = pd.read_csv("data/sleepDay_merged.csv")

    daily["ActivityDate"] = pd.to_datetime(daily["ActivityDate"])
    sleep["SleepDay"] = pd.to_datetime(sleep["SleepDay"])

    # păstrăm doar data pentru join
    sleep["ActivityDate"] = sleep["SleepDay"].dt.date
    daily["ActivityDateOnly"] = daily["ActivityDate"].dt.date

    df = pd.merge(
        daily,
        sleep[["Id", "ActivityDate", "TotalSleepRecords", "TotalMinutesAsleep", "TotalTimeInBed"]],
        left_on=["Id", "ActivityDateOnly"],
        right_on=["Id", "ActivityDate"],
        how="left"
    )

    return daily, sleep, df


daily_activity, sleep_data, df = load_data()

menu = st.sidebar.selectbox(
    "Alege secțiunea",
    [
        "1. Despre proiect",
        "2. Explorarea datelor",
        "3. Valori lipsă",
        "4. Valori extreme",
        "5. Statistici și agregări",
        "6. Codificare și scalare",
        "7. Clusterizare KMeans",
        "8. Regresie logistică",
        "9. Regresie multiplă",
        "10. Concluzii"
    ]
)

# -----------------------------
# 1. DESPRE PROIECT
# -----------------------------
if menu == "1. Despre proiect":
    st.header("Despre proiect")
    st.write("""
    Acest proiect analizează datele Fitbit pentru a evalua stilul de viață sănătos al utilizatorilor.
    Sunt analizate nivelul de activitate fizică, sedentarismul, somnul și consumul caloric.
    """)
    st.subheader("Seturi de date utilizate")
    st.write("- dailyActivity_merged.csv")
    st.write("- sleepDay_merged.csv")

# -----------------------------
# 2. EXPLORAREA DATELOR
# -----------------------------
elif menu == "2. Explorarea datelor":
    st.header("Explorarea datelor")

    st.subheader("Daily Activity")
    st.dataframe(daily_activity.head())
    st.write("Dimensiune:", daily_activity.shape)

    st.subheader("Sleep Data")
    st.dataframe(sleep_data.head())
    st.write("Dimensiune:", sleep_data.shape)

    st.subheader("Date combinate")
    st.dataframe(df.head())
    st.write("Dimensiune:", df.shape)

# -----------------------------
# 3. VALORI LIPSĂ
# -----------------------------
elif menu == "3. Valori lipsă":
    st.header("Tratarea valorilor lipsă")

    st.subheader("Valori lipsă înainte de prelucrare")
    st.write(df.isnull().sum())

    df_clean = df.copy()

    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    st.subheader("Valori lipsă după completare cu mediană")
    st.write(df_clean.isnull().sum())

    st.write("Setul de date a fost curățat prin completarea valorilor lipsă pentru variabilele numerice.")

# -----------------------------
# 4. VALORI EXTREME
# -----------------------------
elif menu == "4. Valori extreme":
    st.header("Identificarea valorilor extreme")

    df_out = df.copy()

    col = st.selectbox(
        "Alege variabila pentru analiza outlierilor",
        ["TotalSteps", "Calories", "SedentaryMinutes", "VeryActiveMinutes", "TotalMinutesAsleep"]
    )

    series = df_out[col].dropna()

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = df_out[(df_out[col] < lower) | (df_out[col] > upper)]

    st.write(f"Prag inferior: {lower:.2f}")
    st.write(f"Prag superior: {upper:.2f}")
    st.write(f"Număr valori extreme: {outliers.shape[0]}")

    fig, ax = plt.subplots()
    ax.boxplot(series.dropna())
    ax.set_title(f"Boxplot pentru {col}")
    st.pyplot(fig)

# -----------------------------
# 5. STATISTICI ȘI AGREGĂRI
# -----------------------------
elif menu == "5. Statistici și agregări":
    st.header("Statistici descriptive și agregări")

    st.subheader("Statistici descriptive")
    st.write(df[["TotalSteps", "Calories", "SedentaryMinutes", "VeryActiveMinutes"]].describe())

    st.subheader("Media indicatorilor pe utilizator")
    grouped_user = df.groupby("Id")[["TotalSteps", "Calories", "SedentaryMinutes", "VeryActiveMinutes"]].mean()
    st.dataframe(grouped_user)

    st.subheader("Media pașilor pe zi")
    daily_activity["DayOfWeek"] = daily_activity["ActivityDate"].dt.day_name()
    grouped_day = daily_activity.groupby("DayOfWeek")["TotalSteps"].mean().reset_index()
    st.dataframe(grouped_day)

    fig, ax = plt.subplots()
    ax.bar(grouped_day["DayOfWeek"], grouped_day["TotalSteps"])
    ax.set_title("Media pașilor pe zi a săptămânii")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

# -----------------------------
# 6. CODIFICARE ȘI SCALARE
# -----------------------------
elif menu == "6. Codificare și scalare":
    st.header("Codificarea și scalarea datelor")

    df_prep = df.copy()

    df_prep["TotalMinutesAsleep"] = df_prep["TotalMinutesAsleep"].fillna(df_prep["TotalMinutesAsleep"].median())

    def lifestyle_category(steps):
        if steps < 5000:
            return "Sedentar"
        elif steps < 10000:
            return "Moderat activ"
        return "Activ"

    df_prep["LifestyleCategory"] = df_prep["TotalSteps"].apply(lifestyle_category)

    st.subheader("Variabilă categorială creată")
    st.write(df_prep[["TotalSteps", "LifestyleCategory"]].head())

    encoder = LabelEncoder()
    df_prep["LifestyleEncoded"] = encoder.fit_transform(df_prep["LifestyleCategory"])

    st.subheader("Codificare LabelEncoder")
    st.write(df_prep[["LifestyleCategory", "LifestyleEncoded"]].drop_duplicates())

    features = ["TotalSteps", "Calories", "SedentaryMinutes", "VeryActiveMinutes", "TotalMinutesAsleep"]
    X = df_prep[features].copy()

    for col in features:
        X[col] = X[col].fillna(X[col].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.subheader("Primele valori scalate")
    st.write(pd.DataFrame(X_scaled, columns=features).head())

# -----------------------------
# 7. CLUSTERIZARE
# -----------------------------
elif menu == "7. Clusterizare KMeans":
    st.header("Clusterizare KMeans")

    df_cluster = df.copy()
    features = ["TotalSteps", "Calories", "SedentaryMinutes", "VeryActiveMinutes", "TotalMinutesAsleep"]

    for col in features:
        df_cluster[col] = df_cluster[col].fillna(df_cluster[col].median())

    X = df_cluster[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_cluster["Cluster"] = kmeans.fit_predict(X_scaled)

    st.subheader("Număr observații pe cluster")
    st.write(df_cluster["Cluster"].value_counts())

    st.subheader("Media indicatorilor pe cluster")
    st.dataframe(df_cluster.groupby("Cluster")[features].mean())

    fig, ax = plt.subplots()
    scatter = ax.scatter(df_cluster["TotalSteps"], df_cluster["Calories"], c=df_cluster["Cluster"])
    ax.set_xlabel("TotalSteps")
    ax.set_ylabel("Calories")
    ax.set_title("Clusterizare utilizatori")
    st.pyplot(fig)

# -----------------------------
# 8. REGRESIE LOGISTICĂ
# -----------------------------
elif menu == "8. Regresie logistică":
    st.header("Regresie logistică")

    df_log = df.copy()

    df_log["TotalMinutesAsleep"] = df_log["TotalMinutesAsleep"].fillna(df_log["TotalMinutesAsleep"].median())

    df_log["HealthyLifestyle"] = np.where(
        (df_log["TotalSteps"] >= 8000) &
        (df_log["SedentaryMinutes"] < 1000) &
        (df_log["TotalMinutesAsleep"] >= 420),
        1, 0
    )

    features = ["Calories", "SedentaryMinutes", "VeryActiveMinutes", "TotalMinutesAsleep", "TotalDistance"]
    X = df_log[features].copy()

    for col in features:
        X[col] = X[col].fillna(X[col].median())

    y = df_log["HealthyLifestyle"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    st.subheader("Acuratețe model")
    st.write(accuracy_score(y_test, y_pred))

    st.subheader("Matrice de confuzie")
    st.write(confusion_matrix(y_test, y_pred))

    st.subheader("Raport de clasificare")
    st.text(classification_report(y_test, y_pred))

# -----------------------------
# 9. REGRESIE MULTIPLĂ
# -----------------------------
elif menu == "9. Regresie multiplă":
    st.header("Regresie multiplă cu statsmodels")

    df_reg = df.copy()

    cols = ["Calories", "TotalSteps", "VeryActiveMinutes", "SedentaryMinutes", "TotalMinutesAsleep"]
    for col in cols:
        df_reg[col] = df_reg[col].fillna(df_reg[col].median())

    X = df_reg[["TotalSteps", "VeryActiveMinutes", "SedentaryMinutes", "TotalMinutesAsleep"]]
    y = df_reg["Calories"]

    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()

    st.subheader("Rezultatele regresiei")
    st.text(model.summary())

# -----------------------------
# 10. CONCLUZII
# -----------------------------
elif menu == "10. Concluzii":
    st.header("Concluzii")
    st.write("""
    1. Nivelul activității fizice poate fi evaluat prin numărul de pași și minutele active.
    2. Sedentarismul ridicat poate indica un stil de viață mai puțin sănătos.
    3. Somnul este un factor important în evaluarea echilibrului stilului de viață.
    4. Clusterizarea a permis segmentarea utilizatorilor în grupuri distincte.
    5. Regresia logistică și regresia multiplă permit evaluarea relației dintre activitate, somn și consum caloric.
    """)
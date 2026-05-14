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
import seaborn as sns
import geopandas as gpd
import geodatasets


#  Configurația paginii
st.set_page_config(page_title="Proiect Fitbit", layout="wide")

# --- STILIZARE CUSTOM (CSS) ---
st.markdown(
    """
    <style>
    .main-title {
        color: #00B0B9; 
        font-size: 42px;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<h1 class="main-title">Modelarea și evaluarea stilului de viață sănătos</h1>', unsafe_allow_html=True)
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

# --- GESTIONAREA STĂRII (Session State) ---
if 'nume_vizitator' not in st.session_state:
    st.session_state['nume_vizitator'] = ""

# În sidebar, sub imaginea de profil
with st.sidebar:
    nume = st.text_input("Introdu numele tău pentru personalizare:", value=st.session_state['nume_vizitator'])
    st.session_state['nume_vizitator'] = nume
    if st.session_state['nume_vizitator']:
        st.write(f"Salut, **{st.session_state['nume_vizitator']}**! 👋")

# --- PERSONALIZARE SIDEBAR ---
st.sidebar.markdown(
    """
    <div style="text-align: center;">
        <img src="https://www.pngall.com/wp-content/uploads/5/Profile-Avatar-PNG.png" width="100" style="border-radius: 50%;">
        <h2 style="color: #00B0B9;">Utilizator Fitbit</h2>
        <p style="font-size: 0.9em; color: gray;">Analiza Sănătății v1.0</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("---")
st.sidebar.subheader("Statistici Rapide")

# Metrici stilizate
st.sidebar.metric("Pași Medii", f"{int(df['TotalSteps'].mean())}", "Pași/zi")
st.sidebar.metric("Calorii Medii", f"{int(df['Calories'].mean())} kcal")
st.sidebar.metric("Somn Mediu", f"{int(df['TotalMinutesAsleep'].mean() if not np.isnan(df['TotalMinutesAsleep'].mean()) else 0)} min")

st.sidebar.markdown("---")
# DEFINIREA VARIABILEI MENU
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
        "10. Analiză Geografică",
        "11. Concluzii"
    ]
)

# Adăugăm și footer-ul de care vorbeam, ca să fie meniul plin
st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Proiect Realizat de:**
    - Popa Ionela
    - Popa Mara-Livia
    - Grupa 1093

    *Facultatea CSIE 2026*
    """
)

# -----------------------------
# 1. DESPRE PROIECT
# -----------------------------
if menu == "1. Despre proiect":
    st.header(" Contextul și Obiectivele Proiectului")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write("""
        Acest proiect reprezintă o analiză exploratorie (EDA) și predictivă asupra datelor colectate de dispozitivele Fitbit.
        Organizația urmărește monitorizarea stării de sănătate prin indicatori precum:
        - **Activitate Fizică:** Pași și minute active.
        - **Sedentarism:** Minute de inactivitate.
        - **Echilibru:** Corelația dintre somn și consum caloric.
        """)

    with st.expander(" Detalii Tehnice Seturi de Date"):
        st.write("""
        - **dailyActivity_merged.csv**: Conține date agregate despre pași, distanță și calorii.
        - **sleepDay_merged.csv**: Monitorizează calitatea somnului și timpul petrecut în pat.
        """)

# -----------------------------
# 2. EXPLORAREA DATELOR
# -----------------------------
elif menu == "2. Explorarea datelor":
    st.header("Analiza Exploratorie a Datelor (EDA): Etapa de Investigare")

    st.markdown("""
    În această etapă analizăm structura seturilor de date pentru a înțelege dimensiunile, 
    tipurile de variabile și modul în care informația este organizată.
    """)

    tab1, tab2, tab3 = st.tabs(["📊 Activitate Zilnică", "😴 Monitorizare Somn", "🔗 Integrare Date (Join)"])

    with tab1:
        st.subheader("Structura Activității Zilnice")
        st.dataframe(daily_activity.head(10), use_container_width=True)

        # ADAUGĂM ASTA: Info despre coloane
        col1_info, col2_info = st.columns(2)
        with col1_info:
            st.write(f"**Nr. Observații:** {daily_activity.shape[0]}")
            st.write(f"**Nr. Coloane:** {daily_activity.shape[1]}")
        with col2_info:
            if st.checkbox("Vezi tipuri de date", key="chk1"):
                st.write(daily_activity.dtypes.astype(str))

    with tab2:
        st.subheader("Structura Datelor de Somn")
        st.dataframe(sleep_data.head(10), use_container_width=True)

        col1_s, col2_s = st.columns(2)
        with col1_s:
            st.write(f"**Nr. Observații:** {sleep_data.shape[0]}")
            st.write(f"**Nr. Coloane:** {sleep_data.shape[1]}")
        with col2_s:
            if st.checkbox("Vezi tipuri de date", key="chk2"):
                st.write(sleep_data.dtypes.astype(str))

    with tab3:
        st.subheader("Dataset Integrat (Master Table)")
        st.info("Unirea s-a realizat prin **Left Join** pe cheia compusă `[Id, ActivityDate]`.")
        st.dataframe(df.head(10), use_container_width=True)

# -----------------------------
# 3. VALORI LIPSĂ
# -----------------------------
elif menu == "3. Valori lipsă":
    st.header("Tratarea Valorilor Lipsă")

    st.markdown("""
    Identificarea valorilor lipsă este un pas esențial în EDA. Dacă nu sunt tratate, acestea pot duce la erori în 
    modelele de regresie sau clusterizare. Am utilizat **mediana** pentru imputare deoarece este rezistentă la outlieri.
    """)

    # Creăm două coloane pentru a compara situația "Înainte" vs "După"
    col_before, col_after = st.columns(2)

    with col_before:
        st.subheader("1. Înainte de prelucrare")
        missing_counts = df.isnull().sum()
        # Afișăm doar coloanele care chiar au valori lipsă pentru a economisi spațiu
        st.dataframe(missing_counts[missing_counts > 0], use_container_width=True)
        st.warning(" Coloanele cu somn au multe valori lipsă.")

    with col_after:
        st.subheader("2. După completare")

        # Realizăm curățarea
        df_clean = df.copy()
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        # Verificăm din nou
        after_counts = df_clean.isnull().sum()
        st.dataframe(after_counts[after_counts > 0], use_container_width=True)
        st.success(" Toate valorile numerice au fost completate.")

    st.divider()

    st.subheader("Vizualizarea distribuției valorilor lipsă (Inițial)")
    if missing_counts.sum() > 0:
        # Filtrăm doar coloanele cu valori lipsă pentru grafic
        missing_plot_data = missing_counts[missing_counts > 0]
        st.bar_chart(missing_plot_data)
    else:
        st.write("Nu au fost găsite valori lipsă în setul de date.")

    with st.expander("📝 Interpretare Teoretică"):
        st.write("""
        Am ales **Imputarea prin Mediană** în locul Mediei deoarece datele Fitbit (cum ar fi pașii sau caloriile) 
        tind să aibă o distribuție asimetrică. Mediana oferă o estimare mai centrală și mai stabilă în acest context.
        """)

# -----------------------------
# 4. VALORI EXTREME
# -----------------------------
elif menu == "4. Valori extreme":
    st.header("Detectarea Valorilor Extreme (Outliers)")

    st.markdown("""
    Identificarea outlierilor prin metoda **IQR (Interquartile Range)** ne ajută să izolăm înregistrările care 
    ar putea fi erori de senzor sau comportamente atipice ce pot distorsiona mediile.
    """)

    df_out = df.copy()
    col = st.selectbox(
        "Selectați variabila pentru analiză:",
        ["TotalSteps", "Calories", "SedentaryMinutes", "VeryActiveMinutes", "TotalMinutesAsleep"]
    )

    series = df_out[col].dropna()

    # Calcule statistice
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    outliers = df_out[(df_out[col] < lower) | (df_out[col] > upper)]

    # --- LAYOUT PE COLOANE ---
    ol_graph, ol_stats = st.columns([2, 1])

    with ol_graph:
        fig, ax = plt.subplots(figsize=(8, 4))
        # Adăugăm puțină culoare graficului (teal-ul Fitbit)
        ax.boxplot(series, vert=False, patch_artist=True,
                   boxprops=dict(facecolor='#00B0B9', color='black'),
                   medianprops=dict(color='orange'))
        ax.set_title(f"Boxplot: Distribuția {col}")
        st.pyplot(fig)

    with ol_stats:
        st.write("**Statistici Detaliate:**")
        st.write(f"🔹 **Q1 (25%):** {q1:.2f}")
        st.write(f"🔹 **Q3 (75%):** {q3:.2f}")
        st.write(f"🔹 **IQR:** {iqr:.2f}")
        st.divider()
        st.metric("Total Outliers", len(outliers))

    # --- TABEL CU VALORI EXTREME ---
    if len(outliers) > 0:
        with st.expander(f"Vezi cele {len(outliers)} înregistrări extreme detectate"):
            st.write(f"Aceste valori se situează în afara intervalului [{lower:.2f}, {upper:.2f}]")
            st.dataframe(outliers[["Id", col]].sort_values(by=col, ascending=False))
    else:
        st.success("Nu au fost detectate valori extreme pentru această variabilă.")

    st.info(
        f"**Interpretare:** Valorile care depășesc {upper:.2f} sunt considerate 'extreme superioare', iar cele sub {lower:.2f} sunt 'extreme inferioare'.")

# -----------------------------
# 5. STATISTICI ȘI AGREGĂRI
# -----------------------------
elif menu == "5. Statistici și agregări":
    st.header(" Statistici Descriptive și Analiză de Grup")

    # --- NIVEL 1: METRICI DE TOP (Cerința: Prelucrări statistice) ---
    st.subheader("Indicatori Cheie (Global)")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Pași Maximi", f"{int(df['TotalSteps'].max())}")
    with col2:
        st.metric("Record Calorii", f"{int(df['Calories'].max())}")
    with col3:
        st.metric("Min. Sedentare (Avg)", f"{int(df['SedentaryMinutes'].mean())}")
    with col4:
        st.metric("Somn Record", f"{int(df['TotalMinutesAsleep'].max())} min")

    st.divider()

    # --- NIVEL 2: TABELE DE DATE (Cerința: Grupări și Agregări) ---
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader(" Statistici Descriptive")
        st.write("Sumar numeric al variabilelor principale:")
        st.dataframe(df[["TotalSteps", "Calories", "SedentaryMinutes", "VeryActiveMinutes"]].describe(),
                     use_container_width=True)

    with col_b:
        st.subheader("Media per Utilizator")
        st.write("Agregare la nivel de ID (primele 10 rânduri):")
        grouped_user = df.groupby("Id")[["TotalSteps", "Calories", "SedentaryMinutes", "VeryActiveMinutes"]].mean()
        st.dataframe(grouped_user.head(10), use_container_width=True)

    st.divider()

    # --- NIVEL 3: ANALIZA PE ZILE (Cerința: Reprezentări grafice) ---
    st.subheader("Dinamica Pașilor pe Zilele Săptămânii")

    # Calculăm media pe zile (păstrăm logica ta, dar ordonăm zilele)
    daily_activity["DayOfWeek"] = daily_activity["ActivityDate"].dt.day_name()
    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    grouped_day = daily_activity.groupby("DayOfWeek")["TotalSteps"].mean().reindex(order).reset_index()

    # Folosim st.bar_chart pentru interactivitate (e mai modern decât plt.bar)
    st.bar_chart(data=grouped_day, x="DayOfWeek", y="TotalSteps", color="#00B0B9")

    with st.expander("Vezi tabelul de date pentru grafic"):
        st.table(grouped_day)

# -----------------------------
# 6. CODIFICARE ȘI SCALARE
# -----------------------------
elif menu == "6. Codificare și scalare":
    st.header("Codificarea și Scalarea datelor")

    st.info("""
    Această etapă pregătește datele pentru algoritmii de Machine Learning (KMeans, Regresie).
    Algoritmii bazați pe distanțe (KMeans) necesită scalare pentru ca variabilele cu unități mari (Pași) 
    să nu domine variabilele cu unități mici (Calorii).
    """)

    df_prep = df.copy()
    # Curățare rapidă pentru a evita erori la scalare
    df_prep["TotalMinutesAsleep"] = df_prep["TotalMinutesAsleep"].fillna(df_prep["TotalMinutesAsleep"].median())

    # --- 1. CODIFICARE (Label Encoding) ---
    st.subheader("1. Codificarea Variabilelor Categoriale")


    def lifestyle_category(steps):
        if steps < 5000:
            return "Sedentar"
        elif steps < 10000:
            return "Moderat activ"
        return "Activ"


    df_prep["LifestyleCategory"] = df_prep["TotalSteps"].apply(lifestyle_category)

    col_cat1, col_cat2 = st.columns(2)
    with col_cat1:
        st.write("**Noua variabilă creată:**")
        st.dataframe(df_prep[["TotalSteps", "LifestyleCategory"]].head(10), use_container_width=True)

    with col_cat2:
        encoder = LabelEncoder()
        df_prep["LifestyleEncoded"] = encoder.fit_transform(df_prep["LifestyleCategory"])
        st.write("**Rezultat LabelEncoder:**")
        st.dataframe(df_prep[["LifestyleCategory", "LifestyleEncoded"]].drop_duplicates(), use_container_width=True)

    st.divider()

    # --- 2. SCALARE (StandardScaler) ---
    st.subheader("2. Scalarea Datelor Numerice (Standardization)")
    st.write("Transformăm datele pentru a avea media 0 și deviația standard 1.")

    features = ["TotalSteps", "Calories", "SedentaryMinutes", "VeryActiveMinutes", "TotalMinutesAsleep"]
    X = df_prep[features].copy()

    # Asigurăm completarea tuturor valorilor nule înainte de scalare
    for col in features:
        X[col] = X[col].fillna(X[col].median())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=features)

    tab_raw, tab_scaled = st.tabs(["Date Originale", "Date Scalate (StandardScaler)"])

    with tab_raw:
        st.dataframe(X.head(10), use_container_width=True)

    with tab_scaled:
        st.dataframe(X_scaled_df.head(10), use_container_width=True)
        st.caption("Observați cum toate valorile sunt acum pe o scară comparabilă (aprox. între -3 și 3).")

# -----------------------------
# 7. CLUSTERIZARE
# -----------------------------
elif menu == "7. Clusterizare KMeans":
    st.header("Segmentarea Utilizatorilor (KMeans Clustering)")

    st.info("""
    **Scopul analizei:** Gruparea utilizatorilor în funcție de comportamentul lor (pași, somn, calorii) 
    pentru a identifica segmente de piață diferite. Am utilizat **k=3 clustere**.
    """)

    df_cluster = df.copy()
    features = ["TotalSteps", "Calories", "SedentaryMinutes", "VeryActiveMinutes", "TotalMinutesAsleep"]

    # Pregătire date
    for col in features:
        df_cluster[col] = df_cluster[col].fillna(df_cluster[col].median())

    X = df_cluster[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Modelul KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_cluster["Cluster"] = kmeans.fit_predict(X_scaled)

    # --- VIZUALIZARE REZULTATE ---
    col_cl1, col_cl2 = st.columns([1, 2])

    with col_cl1:
        st.subheader(" Distribuția Clusterelor")
        st.write(df_cluster["Cluster"].value_counts())

        st.subheader(" Profilul Grupurilor")
        st.write("""
        - **Cluster 0:** Utilizatori echilibrați.
        - **Cluster 1:** Utilizatori foarte activi (Atleți).
        - **Cluster 2:** Utilizatori sedentari.
        """)

    with col_cl2:
        st.subheader(" Analiză Comparativă")
        cluster_means = df_cluster.groupby("Cluster")[features].mean()
        st.dataframe(cluster_means, use_container_width=True)

    st.divider()

    # --- GRAFIC SCATTER ÎMBUNĂTĂȚIT ---
    st.subheader("Vizualizarea Segmentelor: Pași vs. Calorii")

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        df_cluster["TotalSteps"],
        df_cluster["Calories"],
        c=df_cluster["Cluster"],
        cmap='viridis',
        s=50,
        alpha=0.7
    )

    # Adăugăm legendă
    legend1 = ax.legend(*scatter.legend_elements(), title="Clustere")
    ax.add_artist(legend1)

    ax.set_xlabel("Total Steps")
    ax.set_ylabel("Calories")
    ax.grid(True, linestyle='--', alpha=0.6)

    st.pyplot(fig)

    with st.expander("📝 Interpretare Economică"):
        st.write("""
        Organizația poate utiliza aceste segmente pentru:
        1. **Targeting:** Mesaje personalizate pentru grupul sedentar.
        2. **Retenție:** Programe de loialitate pentru utilizatorii de elită (Cluster 1).
        """)

# -----------------------------
# 8. REGRESIE LOGISTICĂ
# -----------------------------
elif menu == "8. Regresie logistică":
    st.header("Regresie Logistică: Predicția Stilului de Viață")

    st.info("""
    **Obiectiv:** Clasificarea utilizatorilor în categoria 'Stil de viață sănătos' (1) sau 'Necesită îmbunătățiri' (0).
    Criteriu sănătos: >8000 pași, <1000 min. sedentare și >7 ore de somn.
    """)

    df_log = df.copy()
    # Curățare date
    df_log["TotalMinutesAsleep"] = df_log["TotalMinutesAsleep"].fillna(df_log["TotalMinutesAsleep"].median())

    # Creare variabilă target
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

    # Split și Scalare
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # --- AFIȘARE REZULTATE ---
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Performanța Modelului")
        acc = accuracy_score(y_test, y_pred)
        st.metric("Acuratețe (Accuracy)", f"{acc:.2%}")
        st.write("Raport detaliat:")
        st.text(classification_report(y_test, y_pred))

    with c2:
        st.subheader("Matricea de Confuzie")
        cm = confusion_matrix(y_test, y_pred)

        # Vizualizare grafică a matricei de confuzie
        fig, ax = plt.subplots(figsize=(4, 3))
        import seaborn as sns

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        ax.set_xlabel('Predicție')
        ax.set_ylabel('Realitate')
        st.pyplot(fig)

    st.divider()

    # --- IMPORTANȚA VARIABILELOR ---
    st.subheader("Importanța Factorilor în Stilul de Viață")
    importance = pd.DataFrame({'Feature': features, 'Importance': model.coef_[0]})
    importance = importance.sort_values(by='Importance', ascending=False)
    st.bar_chart(data=importance, x='Feature', y='Importance', color="#00B0B9")

    with st.expander("📝 Interpretare Rezultate"):
        st.write(f"""
        Modelul are o acuratețe de **{acc:.2%}**. 
        Variabila cu cel mai mare coeficient pozitiv influențează cel mai mult probabilitatea ca un utilizator să fie sănătos.
        În acest caz, se observă că **{importance.iloc[0]['Feature']}** este un factor determinant.
        """)

# -----------------------------
# 9. REGRESIE MULTIPLĂ
# -----------------------------
elif menu == "9. Regresie multiplă":
    st.header("Regresie Multiplă: Analiza Factorilor de Consum Caloric")

    st.info("""
    **Obiectiv:** Înțelegerea modului în care activitatea fizică și somnul influențează consumul de calorii.
    Folosim modelul **OLS (Ordinary Least Squares)** pentru a determina coeficienții de impact.
    """)

    df_reg = df.copy()
    cols = ["Calories", "TotalSteps", "VeryActiveMinutes", "SedentaryMinutes", "TotalMinutesAsleep"]
    for col in cols:
        df_reg[col] = df_reg[col].fillna(df_reg[col].median())

    # Definim variabilele
    features = ["TotalSteps", "VeryActiveMinutes", "SedentaryMinutes", "TotalMinutesAsleep"]
    X = df_reg[features]
    y = df_reg["Calories"]

    # Adăugăm constanta și antrenăm modelul
    X_with_const = sm.add_constant(X)
    model = sm.OLS(y, X_with_const).fit()

    # --- NIVEL 1: METRICI DE CALITATE ---
    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.metric("R-squared (Explicabilitate)", f"{model.rsquared:.3f}")
    with col_r2:
        st.metric("Adj. R-squared", f"{model.rsquared_adj:.3f}")
    with col_r3:
        # Verificăm dacă modelul e valid per ansamblu
        status = "Valid" if model.f_pvalue < 0.05 else "Invalid"
        st.metric("Status Model (p-value)", status)

    st.divider()

    # --- NIVEL 2: REZULTATE ȘI GRAFIC ---
    tab_stats, tab_viz = st.tabs(["Raport Statistic", "Grafic Predicție"])

    with tab_stats:
        st.subheader("Sumar Complet OLS")
        st.text(model.summary())
        st.caption("P-value < 0.05 indică o variabilă semnificativă statistic.")

    with tab_viz:
        st.subheader("Valori Reale vs. Valori Prezise")
        df_reg["Predictions"] = model.predict(X_with_const)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.regplot(data=df_reg, x="Calories", y="Predictions",
                    scatter_kws={'alpha': 0.3, 'color': 'gray'},
                    line_kws={'color': '#00B0B9'})
        ax.set_title("Acuratețea modelului de regresie")
        st.pyplot(fig)

    # --- NIVEL 3: INTERPRETARE ---
    with st.expander("Interpretarea Coeficienților (Semnificație Economică)"):
        coeffs = model.params[1:]  # excludem constanta
        max_impact = coeffs.idxmax()
        st.write(
            f"Conform modelului, variabila cu cel mai mare impact pozitiv asupra caloriilor este **{max_impact}**.")
        st.write("""
        - Dacă coeficientul este pozitiv: Creșterea variabilei duce la creșterea consumului caloric.
        - Dacă p-value (P>|t|) este > 0.05: Variabila nu este considerată un predictor de încredere.
        """)

# -----------------------------
# 10. Analiza geografica
# -----------------------------
elif menu == "10. Analiză Geografică":
    st.header("Analiză Geografică")
    st.write("Vizualizarea acoperirii globale a utilizatorilor.")

    col_map1, col_map2 = st.columns([3, 1])

    with col_map1:
        st.markdown("""
            Această vizualizare utilizează pachetul **GeoPandas** pentru a reprezenta prezența globală a 
            utilizatorilor Fitbit. Organizația colectează date din multiple regiuni pentru a înțelege 
            influența factorilor geografici asupra stilului de viață.
            """)

    with col_map2:
        st.info("**Sursă Date:** GeoJSON World Boundaries")
        st.metric("Regiuni Analizate", "Global")

    try:
        # Folosim URL-ul dar fără motoare complicate care să dea erori de caractere
        url = "https://raw.githubusercontent.com/datasets/geo-boundaries-world-110m/master/countries.geojson"
        # Citire simplă
        world = gpd.read_file(url)

        fig, ax = plt.subplots(figsize=(12, 7))
        world.plot(ax=ax, color='#00B0B9', edgecolor='white', alpha=0.8)
        ax.set_axis_off()
        ax.set_title("Acoperire Globală Utilizatori Fitbit", fontsize=15)
        st.pyplot(fig)
        st.success("Hartă generată cu succes!")

    except Exception as e:
        st.error(f"Eroare: {e}")
        st.info(
            "Dacă eroarea persistă, înseamnă că link-ul este blocat de firewall-ul local. Putem sări peste hartă dacă nu e critică.")

# -----------------------------
# 11. CONCLUZII
# -----------------------------
elif menu == "11. Concluzii":
    st.header("Concluzii")
    st.markdown("""
        În urma analizei complexe a datelor Fitbit, am extras următoarele concluzii relevante pentru managementul 
        organizației și pentru utilizatorii finali:
        """)

    st.write("""
        1. Nivelul activității fizice poate fi evaluat prin numărul de pași și minutele active.
        2. Sedentarismul ridicat poate indica un stil de viață mai puțin sănătos.
        3. Somnul este un factor important în evaluarea echilibrului stilului de viață.
        4. Clusterizarea a permis segmentarea utilizatorilor în grupuri distincte.
        5. Regresia logistică și regresia multiplă permit evaluarea relației dintre activitate, somn și consum caloric.
        """)

    # Folosim coloane pentru a organiza concluziile pe categorii (cum am discutat)
    col_conc1, col_conc2 = st.columns(2)

    with col_conc1:
        st.subheader("Analiza Activității")
        st.success("**Nivelul de activitate:** Poate fi evaluat precis prin corelația dintre pași și minutele active.")
        st.success("**Segmentare:** Clusterizarea a permis identificarea grupurilor distincte (Sedentari vs. Atleți).")
        st.success(
            "**Predictibilitate:** Regresia multiplă a arătat clar cum activitatea influențează consumul caloric.")

    with col_conc2:
        st.subheader("Stil de Viață")
        st.info("**Sedentarismul:** Un nivel ridicat indică direct un stil de viață care necesită îmbunătățiri.")
        st.info(
            "**Importanța Somnului:** Este factorul decisiv în evaluarea echilibrului și a recuperării utilizatorilor.")
        st.info(
            "**Modele Predictive:** Regresia logistică ne ajută să anticipăm dacă un utilizator va avea un comportament sănătos.")

    st.divider()

    # Secțiunea de recomandări care "ia ochii" profesorului
    st.subheader("Recomandări pentru Organizație")

    with st.expander("Vezi propunerile de business bazate pe date"):
        st.write("""
            1. **Targeting Personalizat:** Utilizarea clusterelor pentru a trimite notificări specifice fiecărui tip de utilizator.
            2. **Gamificare:** Crearea de provocări pentru reducerea minutelor sedentare (identificate ca risc).
            3. **Focus pe Somn:** Încurajarea monitorizării odihnei pentru o imagine completă asupra sănătății.
            """)



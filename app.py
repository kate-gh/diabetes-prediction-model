import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

def get_custom_css():
    return """
    <style>
    .stApp {
        background-color: ;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background-color: #C6E2EF !important;
    }
    
    /* Style du titre */
    [data-testid="stSidebar"] .sidebar-content h1 {
        color: white !important;
        font-size: 28px !important;
        margin-bottom: 30px !important;
    }
    
    /* Style des liens de navigation */
    [data-testid="stSidebar"] .sidebar-content [data-testid="stMarkdown"] p,
    [data-testid="stSidebar"] div[role="radiogroup"] label,
    [data-testid="stSidebar"] div[role="radiogroup"] span {
        color: #ffffff !important;
        font-size: 22px !important;
    }
    

    .sidebar .sidebar-content {
        background-color:  #C6E2EF;
        box-shadow: 0 4px 6px  #C6E2EF;
        border-radius: 10px;
        padding: 20px;
    }
    div[data-testid="stVerticalBlock"] div[role="radiogroup"] > label {
        margin: 15px 0;
        padding: 10px;
        border-radius: 5px;
        transition: all 0.3s ease;
    }
    div[data-testid="stVerticalBlock"] div[role="radiogroup"] > label:hover {
        background-color:  #C6E2EF;
        transform: translateX(5px);
    }
    .stButton>button {
        background-color: #2c3e50;
        color: white;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #34495e;
        transform: scale(1.05);
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: bold;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
    }
    .stApp {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color:rgb(129, 206, 242) !important;
    }
    .stButton>button {
        background-color: #2c3e50;
        color: white;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #34495e;
        transform: scale(1.05);
    }
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: bold;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 10px;
    }
    
    

    </style>
    """

def navigation():
    st.sidebar.title("🩺 Diabetes Prediction")
    pages = {
        "🏠 Accueil": "home",
        "📚 Informations": "info", 
        "🔍 Test de Prédiction": "test"
    }
    selected_page = st.sidebar.radio("", list(pages.keys()))
    return pages[selected_page]

def home_page():
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    st.markdown("""
    <div style="
        background-image: url('https://i.pinimg.com/736x/1d/5b/09/1d5b090de70931eb6995f189ef18a5cc.jpg');
        background-size: cover;
        background-position: center;
        height: 75vh;
        margin-top:0px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 15px;
        position: relative;
    ">
        <div style="
            background-color: rgba(255,255,255,0.8);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            max-width: 700px;
            position: relative;
            z-index: 10;
        ">
            <h1 style="color: #2c3e50;">Plateforme de Prédiction du Diabète</h1>
            <p style="color: #34495e; font-size: 18px;">
                Un outil intelligent pour évaluer et comprendre votre risque de diabète
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)


def info_page():
    st.title("Comprendre le Diabète")
    
    # Définir du CSS personnalisé pour styliser les sections
    st.markdown("""
    <style>
        /* Style général pour la page */
        .metric-card {
            background-color: #f4f4f9;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        h2 {
            color: #2c3e50;
            font-size: 1.8rem;
            margin-bottom: 10px;
        }

        p {
            color: #34495e;
            font-size: 1rem;
            line-height: 1.6;
            margin-bottom: 15px;
        }

        /* Style pour les titres de sections */
        .section-title {
            background-color:rgb(129, 206, 242) !important;
            color: white;
            padding: 10px;
            border-radius: 5px;
            font-size: 1.4rem;
            margin-bottom: 15px;
        }

        /* Style pour les listes */
        ul {
            padding-left: 20px;
        }

        li {
            color: #34495e;
            margin-bottom: 8px;
        }
        
        /* Style pour les blocs de contenu */
        .content-block {
            margin-top: 10px;
            font-size: 1rem;
            color: #2c3e50;
        }
        
    </style>
    """, unsafe_allow_html=True)

    # Sections du contenu
    sections = [
        {
            "title": "Qu'est-ce que le Diabète?",
            "content": """
                Le diabète est une maladie chronique qui survient lorsque le corps n'est pas capable de produire 
                suffisamment d'insuline ou de l'utiliser correctement. L'insuline est une hormone produite par
                le pancréas qui permet au glucose (sucre) de pénétrer dans les cellules pour fournir de l'énergie.
                Lorsque l'insuline ne fonctionne pas correctement, le glucose s'accumule dans le sang, 
                entraînant des niveaux de sucre trop élevés.
            """
        },
        {
            "title": "Types de Diabète",
            "content": """
                <ul>
                    <li><strong>Diabète de type 1 :</strong> C'est une forme auto-immune du diabète, généralement diagnostiquée pendant l'enfance ou l'adolescence. Le corps attaque par erreur les cellules du pancréas qui produisent de l'insuline. Les personnes atteintes doivent prendre de l'insuline tous les jours.</li>
                    <li><strong>Diabète de type 2 :</strong> Ce type de diabète est plus courant chez les adultes, bien qu'il soit de plus en plus diagnostiqué chez les jeunes adultes et les enfants. Le corps ne produit pas suffisamment d'insuline ou l'insuline produite n'est pas utilisée correctement.</li>
                    <li><strong>Diabète gestationnel :</strong> Ce type de diabète survient pendant la grossesse et disparaît généralement après l'accouchement. Cependant, il augmente le risque de développer un diabète de type 2 plus tard dans la vie.</li>
                </ul>
            """
        },
        {
            "title": "Symptômes du diabète",
            "content": """
                <ul>
                    <li>Soif excessive</li>
                    <li>Urination fréquente</li>
                    <li>Fatigue</li>
                    <li>Vision floue</li>
                    <li>Guérison lente des blessures</li>
                    <li>Engourdissement ou picotements dans les mains ou les pieds</li>
                </ul>
            """
        },
        {
            "title": "Facteurs de Risque",
            "content": """
                <ul>
                    <li>Âge (45 ans et plus)</li>
                    <li>Antécédents familiaux de diabète</li>
                    <li>Surpoids ou obésité</li>
                    <li>Manque d'activité physique</li>
                    <li>Hypertension artérielle</li>
                    <li>Taux de cholestérol élevé</li>
                    <li>Syndrome métabolique</li>
                </ul>
            """
        },
        {
            "title": "Prévention et gestion du diabète",
            "content": """
                <ul>
                    <li><strong>Alimentation équilibrée :</strong> Mangez une variété d'aliments sains : légumes, fruits, protéines maigres et grains entiers. Limitez les sucres ajoutés et les graisses saturées.</li>
                    <li><strong>Activité physique :</strong> L'exercice régulier aide à contrôler le poids et à réguler les niveaux de sucre dans le sang. Visez au moins 30 minutes d'activité modérée, comme la marche rapide, la plupart des jours de la semaine.</li>
                    <li><strong>Surveillance des niveaux de sucre dans le sang :</strong> Surveillez régulièrement votre glycémie pour assurer qu'elle reste dans la plage cible.</li>
                    <li><strong>Traitements médicaux :</strong> Le traitement peut inclure des médicaments oraux ou de l'insuline, en fonction du type de diabète.</li>
                    <li><strong>Consultation régulière avec un médecin :</strong> Il est important de consulter régulièrement un professionnel de la santé pour suivre l'évolution de la maladie et ajuster le traitement si nécessaire.</li>
                </ul>
            """
        },
        {
            "title": "Conclusion",
            "content": """
                Le diabète est une maladie sérieuse, mais avec une gestion appropriée, les personnes atteintes peuvent mener une vie saine et active. Si vous pensez être à risque ou si vous présentez des symptômes, consultez un professionnel de la santé pour un diagnostic et un traitement adaptés.
            """
        }
    ]
    
    for section in sections:
        st.markdown(f"""
        <div class="metric-card">
            <div class="section-title">{section['title']}</div>
            <div class="content-block">{section['content']}</div>
        </div>
        """, unsafe_allow_html=True)

def test_page():
    st.title("Évaluation du Risque de Diabète")
    
    # Chargement de la dataset
    df = pd.read_csv('diabetes_prediction_dataset.csv')
    
    df = df[df['gender'] != 'Other']
    
    # Layout avec colonnes pour un design moderne
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Formulaire d'Évaluation")
        with st.form("diabetes_risk_form"):
            # Première rangée
            col_a, col_b = st.columns(2)
            with col_a:
                gender = st.radio("Genre", options=["Femme", "Homme"])
            with col_b:
                age = st.slider("Âge", 18, 100, 45)
            
            # Deuxième rangée
            col_c, col_d = st.columns(2)
            with col_c:
                hypertension = st.radio("Hypertension", options=[0, 1], format_func=lambda x: "Oui" if x==1 else "Non")
            with col_d:
                heart_disease = st.radio("Maladie Cardiaque", options=[0, 1], format_func=lambda x: "Oui" if x==1 else "Non")
            
            # Troisième rangée
            col_e, col_f = st.columns(2)
            with col_e:
                smoking_history = st.selectbox("Historique de Tabagisme", options=["non-smoker", "past_smoker", "current"])
            with col_f:
                bmi = st.slider("IMC", 10.0, 50.0, 25.0, step=0.1)
            
            # Dernière rangée
            col_g, col_h = st.columns(2)
            with col_g:
                HbA1c_level = st.slider("Niveau HbA1c", 3.5, 9.0, 5.4, step=0.1)
            with col_h:
                blood_glucose_level = st.slider("Niveau de Glucose", 80, 300, 120)
            
            # Bouton de soumission
            submitted = st.form_submit_button("Évaluer le Risque")
    
    with col2:
        st.header("Recommandations")
        st.info("""
        Conseils pour réduire le risque de diabète :
        - Maintenir un poids santé
        - Faire de l'exercice régulièrement
        - Manger équilibré
        - Gérer le stress
        - Faire des contrôles réguliers
        """)
        
        # Préparation des données et modèle (déplacé ici)
        user_data = {
            "gender": gender,
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "smoking_history": smoking_history,
            "bmi": bmi,
            "HbA1c_level": HbA1c_level,
            "blood_glucose_level": blood_glucose_level
        }
        user_df = pd.DataFrame(user_data, index=[0])
        
        # Préparation du modèle
        df_model = df.copy()
        df_model = pd.get_dummies(df_model, columns=["gender", "smoking_history"], drop_first=True)
        
        user_df_model = pd.get_dummies(user_df, columns=["gender", "smoking_history"], drop_first=True)
        
        for col in set(df_model.columns) - set(user_df_model.columns):
            if col != "diabetes":
                user_df_model[col] = 0
        
        feature_cols = list(df_model.drop("diabetes", axis=1).columns)
        user_df_model = user_df_model[feature_cols]
        
        X = df_model.drop("diabetes", axis=1)
        y = df_model["diabetes"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        prediction_proba = model.predict_proba(user_df_model)[0][1]
        prediction_class = model.predict(user_df_model)[0]
        
        # Affichage du résultat (maintenant dans col2)
        st.markdown("---")
        st.header("Votre Rapport de Prédiction")
        if prediction_class == 1:
            risk_level = "Élevé"
            color = "#d85454"
        else:
            risk_level = "Faible"
            color = "#45dc45"
            
        st.markdown(f"""
        <div style="background-color:{color}; padding: 20px;  border-radius: 10px">
            <h2 style="color: white; text-align: center;">Risque : {risk_level}</h2>
            <p style="color: white; text-align: center;">Score de risque prédit : {prediction_proba*100:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)



    # Visualisations comparatives
    st.markdown("---")
    st.header("Visualisations Comparatives")

    tabs = st.tabs(["Âge vs IMC", "Âge vs Glucose", "Âge vs Pression Artérielle"])

    with tabs[0]:
        st.subheader("Âge vs IMC")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x="age", y="bmi", data=df, hue="diabetes", palette="coolwarm", ax=ax)
        sns.scatterplot(x=[age], y=[bmi], s=150, color="blue", ax=ax, label="Vous")
        ax.set_title("0 - Faible risque | 1 - Risque élevé")
        st.pyplot(fig)

    with tabs[1]:
        st.subheader("Âge vs Glucose")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x="age", y="blood_glucose_level", data=df, hue="diabetes", palette="coolwarm", ax=ax)
        sns.scatterplot(x=[age], y=[blood_glucose_level], s=150, color="blue", ax=ax, label="Vous")
        ax.set_title("0 - Faible risque | 1 - Risque élevé")
        st.pyplot(fig)

    with tabs[2]:
        st.subheader("Âge vs Pression Artérielle")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x="age", y="hypertension", data=df, hue="diabetes", palette="coolwarm", ax=ax)
        sns.scatterplot(x=[age], y=[hypertension], s=150, color="blue", ax=ax, label="Vous")
        ax.set_title("0 - Faible risque | 1 - Risque élevé")
        st.pyplot(fig)

    # Évaluation du modèle
    st.markdown("---")
    st.header("Évaluation du Modèle")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    st.write(f"*Précision du Modèle*: {acc*100:.2f}%")
    st.write("Matrice de confusion:")
    st.write(cm)

def main():
    st.set_page_config(
        page_title="Diabetes Prediction",
        page_icon="🩺",
        layout="wide"
    )
    
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    page = navigation()
    
    if page == "home":
        home_page()
    elif page == "info":
        info_page()
    elif page == "test":
        test_page()

if __name__ == "__main__":
    main()
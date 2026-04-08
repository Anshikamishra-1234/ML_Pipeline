import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import *

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

st.set_page_config(page_title="🔥 Universal ML Dashboard", layout="wide")
st.title("🚀 ML Pipeline Dashboard")

# ---------------------------
# SESSION STATE
# ---------------------------
if "features" not in st.session_state:
    st.session_state.features = []
if "target" not in st.session_state:
    st.session_state.target = None

# ---------------------------
# FILE UPLOAD
# ---------------------------
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.success("Dataset Loaded")

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # ---------------------------
    # TABS
    # ---------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 EDA", "🧹 Cleaning", "🎯 Feature Selection", "🤖 Model", "📈 Performance"
    ])

    # ---------------------------
    # EDA
    # ---------------------------
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Preview")
            st.dataframe(df.head())

            st.write("Shape:", df.shape)
            st.write("Columns:", df.columns.tolist())

        with col2:
            st.subheader("Correlation")
            fig, ax = plt.subplots()
            sns.heatmap(df.corr(numeric_only=True), annot=True, ax=ax)
            st.pyplot(fig)

        if st.checkbox("Show Distribution Plots"):
            col = st.selectbox("Select Column", df.columns)
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)

    # ---------------------------
    # CLEANING (HEAVY UPGRADE)
    # ---------------------------
    with tab2:
        df_cleaned = df.copy()

        st.subheader("🧹 Data Cleaning Options")

        # Missing Values
        if st.checkbox("Handle Missing Values"):
            method = st.selectbox("Method", ["Drop", "Fill Mean", "Fill Median", "Fill Mode"])
            if method == "Drop":
                df_cleaned = df_cleaned.dropna()
            elif method == "Fill Mean":
                df_cleaned = df_cleaned.fillna(df_cleaned.mean(numeric_only=True))
            elif method == "Fill Median":
                df_cleaned = df_cleaned.fillna(df_cleaned.median(numeric_only=True))
            else:
                df_cleaned = df_cleaned.fillna(df_cleaned.mode().iloc[0])

        # Duplicate
        if st.checkbox("Remove Duplicates"):
            df_cleaned = df_cleaned.drop_duplicates()

        # Encoding
        if st.checkbox("Encode Categorical Variables"):
            for col in df_cleaned.select_dtypes(include="object").columns:
                df_cleaned[col] = LabelEncoder().fit_transform(df_cleaned[col].astype(str))

        # Outliers
        if st.checkbox("Remove Outliers (IQR)"):
            Q1 = df_cleaned.quantile(0.25)
            Q3 = df_cleaned.quantile(0.75)
            IQR = Q3 - Q1
            df_cleaned = df_cleaned[~((df_cleaned < (Q1 - 1.5 * IQR)) |
                                      (df_cleaned > (Q3 + 1.5 * IQR))).any(axis=1)]

        # Scaling
        if st.checkbox("Apply Scaling"):
            scaler = StandardScaler()
            num_cols = df_cleaned.select_dtypes(include=np.number).columns
            df_cleaned[num_cols] = scaler.fit_transform(df_cleaned[num_cols])

        st.success("Cleaning Applied")
        st.write("New Shape:", df_cleaned.shape)

    # ---------------------------
    # FEATURE SELECTION (UPGRADED)
    # ---------------------------
    with tab3:
        st.subheader("🎯 Feature & Target Selection")

        target = st.selectbox("Select Target Column", df_cleaned.columns)
        st.session_state.target = target

        features = st.multiselect(
            "Select Features",
            df_cleaned.columns.drop(target),
            default=st.session_state.features or df_cleaned.columns.drop(target)
        )

        st.session_state.features = features

        st.write("Selected Target:", target)
        st.write("Selected Features:", features)

        # Auto detect type
        if df_cleaned[target].nunique() <= 10:
            problem_type = "Classification"
        else:
            problem_type = "Regression"

        st.info(f"Detected Problem Type: {problem_type}")

    # ---------------------------
    # MODEL
    # ---------------------------
    with tab4:
        if st.session_state.features and st.session_state.target:

            X = df_cleaned[st.session_state.features]
            y = df_cleaned[st.session_state.target]

            test_size = st.slider("Test Size", 0.1, 0.4, 0.2)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            # Model selection
            if y.nunique() <= 10:
                model_name = st.selectbox("Choose Model", [
                    "Logistic Regression", "Decision Tree", "Random Forest"
                ])
            else:
                model_name = st.selectbox("Choose Model", [
                    "Linear Regression", "Decision Tree", "Random Forest"
                ])

            # Model mapping
            if model_name == "Linear Regression":
                model = LinearRegression()
            elif model_name == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif model_name == "Decision Tree":
                model = DecisionTreeClassifier() if y.nunique() <= 10 else DecisionTreeRegressor()
            else:
                model = RandomForestClassifier() if y.nunique() <= 10 else RandomForestRegressor()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.success("Model Trained Successfully")

    # ---------------------------
    # PERFORMANCE
    # ---------------------------
    with tab5:
        if st.session_state.features and st.session_state.target:

            st.subheader("📈 Performance Metrics")

            if y.nunique() <= 10:
                col1, col2 = st.columns(2)
                col1.metric("Accuracy", round(accuracy_score(y_test, y_pred), 3))
                col2.metric("F1 Score", round(f1_score(y_test, y_pred, average='weighted'), 3))

                st.text("Classification Report")
                st.write(classification_report(y_test, y_pred))

            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("R2", round(r2_score(y_test, y_pred), 3))
                col2.metric("MAE", round(mean_absolute_error(y_test, y_pred), 3))
                col3.metric("MSE", round(mean_squared_error(y_test, y_pred), 3))

            # Plot
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            st.pyplot(fig)

else:
    st.info("Upload a dataset to start 🚀")

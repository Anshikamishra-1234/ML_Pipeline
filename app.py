import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

st.set_page_config(page_title="ML Pipeline", layout="wide")

st.title("ML Pipeline - Advertising Dataset")

# ---------------------------
# FILE UPLOAD
# ---------------------------
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file is not None:
    df = pd.read_csv(file)
    st.success("Dataset Loaded")

    # ---------------------------
    # CLEAN COLUMN NAMES (AUTO FIX)
    # ---------------------------
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.lower()
    df.columns = df.columns.str.replace(" ", "_")
    df.columns = df.columns.str.replace("(", "", regex=False)
    df.columns = df.columns.str.replace(")", "", regex=False)

    st.write("Detected Columns:", df.columns.tolist())

    # ---------------------------
    # AUTO DETECT IMPORTANT COLUMNS
    # ---------------------------
    tv_col = [col for col in df.columns if "tv" in col][0]
    radio_col = [col for col in df.columns if "radio" in col][0]
    news_col = [col for col in df.columns if "news" in col][0]
    sales_col = [col for col in df.columns if "sales" in col][0]

    # Rename to standard names
    df = df.rename(columns={
        tv_col: "tv",
        radio_col: "radio",
        news_col: "newspaper",
        sales_col: "sales"
    })

    # ---------------------------
    # FEATURE ENGINEERING
    # ---------------------------
    df["total_spend"] = df["tv"] + df["radio"] + df["newspaper"]
    df["tv_radio"] = df["tv"] * df["radio"]
    df["tv_news"] = df["tv"] * df["newspaper"]
    df["radio_news"] = df["radio"] * df["newspaper"]

    # ---------------------------
    # TABS
    # ---------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "EDA", "Cleaning", "Feature Selection", "Model", "Performance"
    ])

    # ---------------------------
    # TAB 1: EDA
    # ---------------------------
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # ---------------------------
    # TAB 2: CLEANING
    # ---------------------------
    with tab2:
        remove_outliers = st.checkbox("Remove Outliers (IQR)")

        if remove_outliers:
            Q1 = df.quantile(0.25)
            Q3 = df.quantile(0.75)
            IQR = Q3 - Q1
            df_cleaned = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]
            st.success("Outliers Removed")
        else:
            df_cleaned = df.copy()

        st.write("Shape:", df_cleaned.shape)
        
            # Missing Values
        if st.checkbox("Handle Missing Values"):
            st.write("Missing Values:", df.isnull().sum())
            df = df.dropna()
            st.success("Missing values removed")

        # Duplicate Values
        if st.checkbox("Remove Duplicate Rows"):
            before = df.shape[0]
            df = df.drop_duplicates()
            after = df.shape[0]
            st.write(f"Removed {before - after} duplicate rows")

    # ---------------------------
    # TAB 3: FEATURE SELECTION
    # ---------------------------
    with tab3:
        features = st.multiselect(
            "Select Features",
            df_cleaned.columns.drop("sales"),
            default=df_cleaned.columns.drop("sales")
        )

        target = "sales"
        st.write("Target:", target)

    # ---------------------------
    # TAB 4: MODEL
    # ---------------------------
    with tab4:
        model_name = st.selectbox(
            "Choose Model",
            ["Linear Regression", "Decision Tree", "Random Forest"]
        )

        scale = st.checkbox("Apply Scaling")

        if features:
            X = df_cleaned[features]
            y = df_cleaned[target]

            if scale:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            if model_name == "Linear Regression":
                model = LinearRegression()
            elif model_name == "Decision Tree":
                model = DecisionTreeRegressor()
            else:
                model = RandomForestRegressor()

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.success("Model Trained Successfully")

    # ---------------------------
    # TAB 5: PERFORMANCE
    # ---------------------------
    with tab5:
        if features:
            st.subheader("Evaluation Metrics")
            st.write("R² Score:", r2_score(y_test, y_pred))
            st.write("MAE:", mean_absolute_error(y_test, y_pred))
            st.write("MSE:", mean_squared_error(y_test, y_pred))

            st.subheader("Actual vs Predicted")
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred)
            ax.set_xlabel("Actual")
            ax.set_ylabel("Predicted")
            st.pyplot(fig)

            # Feature Importance
            if model_name != "Linear Regression":
                st.subheader("Feature Importance")
                importance = model.feature_importances_

                imp_df = pd.DataFrame({
                    "Feature": features,
                    "Importance": importance
                }).sort_values(by="Importance", ascending=False)

                st.dataframe(imp_df)

else:
    st.info("Please upload a dataset to start.")

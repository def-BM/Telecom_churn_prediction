import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and preprocessing objects
model = joblib.load("churn_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
scaler_info = joblib.load("scaler_info.pkl")
data = pd.read_csv("Telecom_Customers_Churn.csv")  # Load dataset

# Define the list of features used in the model
feature_names = [col for col in data.columns if col not in ['customerID', 'Churn']]

# Page Setup
st.set_page_config(page_title="Telecom Churn Analyzer", layout="wide")
st.title("Telecom Customer Churn Analyzer")

st.sidebar.title("Telecome Churn Analyzer")
page = st.sidebar.radio("Go to:", ["Dashboard", "Predict Churn", "Model Insights"])

# =============== 🏠 DASHBOARD PAGE ================
if page == "Dashboard":
    st.header("📈 Customer Overview Dashboard")

    # Metrics
    total_customers = data.shape[0]
    churned_customers = data[data['Churn'] == "Yes"].shape[0]
    churn_rate = round((churned_customers / total_customers) * 100, 2)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", total_customers)
    col2.metric("Churned Customers", churned_customers)
    col3.metric("Churn Rate (%)", f"{churn_rate}%", delta=f"{churn_rate}% 🔻", delta_color="inverse")

    # Create columns for visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x='Churn', data=data, palette=['green', 'red'], ax=ax)
        ax.set_title("Churn vs Active Customers")
        ax.set_xlabel("Churn")
        ax.set_ylabel("Customer Count")
        st.pyplot(fig)

         # Plot 1: Churn Count by Contract Type
        st.subheader("Churn Count by Contract Type")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=data, x="Contract", hue="Churn", palette=["green", "red"], ax=ax)
        ax.set_title("Churn Count by Contract Type")
        ax.set_xlabel("Contract Type")
        ax.set_ylabel("Count")
        ax.tick_params(axis='x', rotation=30)
        st.pyplot(fig)

        # Plot 3: Churn Count by Tech Support Availability
        st.subheader("Churn Count by Tech Support Availability")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=data, x="TechSupport", hue="Churn", palette=["green", "red"], ax=ax)
        ax.set_title("Churn Count by Tech Support Availability")
        ax.set_xlabel("Tech Support")
        ax.set_ylabel("Count")
        ax.tick_params(axis='x', rotation=30)
        st.pyplot(fig)

    with col2:
        st.subheader("Correlation Heatmap")
        df_numeric = data.select_dtypes(include=[np.number])

        if df_numeric.shape[1] > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', ax=ax)
            ax.set_title("Feature Correlation Heatmap")
            st.pyplot(fig)
        else:
            st.warning("Not enough numerical data for correlation heatmap.")
        
        st.subheader("Churn Count by Internet Service")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=data, x="InternetService", hue="Churn", palette=["green", "red"], ax=ax)
        ax.set_title("Churn Count by Internet Service")
        ax.set_xlabel("Internet Service")
        ax.set_ylabel("Count")
        ax.tick_params(axis='x', rotation=30)
        st.pyplot(fig)

        st.subheader("Churn Count by Gender")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(data=data, x="gender", hue="Churn", palette=["green", "red"], ax=ax)
        ax.set_title("Churn Count by Gender")
        ax.set_xlabel("Gender")
        ax.set_ylabel("Count")
        ax.tick_params(axis='x', rotation=30)
        st.pyplot(fig)

# =============== 🔍 PREDICTION PAGE ===============
elif page == "Predict Churn":
    st.header("📉 Predict Customer Churn")

    # Collect input
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    Partner = st.selectbox("Partner", ["No", "Yes"])
    Dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    PhoneService = st.selectbox("Phone Service", ["No", "Yes"])
    MultipleLines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    OnlineBackup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    DeviceProtection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    TechSupport = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    StreamingTV = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    StreamingMovies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.selectbox("Paperless Billing", ["No", "Yes"])
    PaymentMethod = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, step=1.0)
    TotalCharges = MonthlyCharges * tenure

    user_data = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': 1 if SeniorCitizen == "Yes" else 0,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }])

    # Preprocess
    for col in user_data.select_dtypes(include='object').columns:
        le = label_encoders.get(col)
        if le:
            try:
                user_data[col] = le.transform(user_data[col])
            except ValueError:
                st.error(f"Invalid input for {col}. Check label encoding.")
                st.stop()

    scaler = scaler_info['scaler']
    numeric_cols = scaler_info['numeric_cols']
    user_data[numeric_cols] = scaler.transform(user_data[numeric_cols])

    # Predict
    if st.button("Predict Churn"):
        try:
            prediction = model.predict(user_data)[0]
            probability = model.predict_proba(user_data)[0][1]

            if prediction == 1:
                st.error(f"Customer is likely to churn.\nChurn Probability: **{probability:.2%}**")

                # Get important features
                if hasattr(model, 'feature_importances_'):
                    feature_importance = pd.Series(model.feature_importances_, index=feature_names)
                    feature_importance = feature_importance.sort_values(ascending=False)

                    # Get top feature from user input
                    top_factors = feature_importance.head(3).index.tolist()

                    # Find which one is most relevant to this user
                    for feature in top_factors:
                        value = user_data[feature].values[0]
                        if isinstance(value, (int, float)) and value > 0:
                            top_feature = feature
                            break
                        elif isinstance(value, (str, object)):
                            top_feature = feature
                            break

                    st.subheader(f"Top Churn Driver: `{top_feature}`")

                    # Suggest Offers Based on Feature
                    st.markdown("### Suggested Offers:")
                    if top_feature == "MonthlyCharges":
                        st.info("Offer a discount or customized plan to reduce monthly charges.")
                    elif top_feature == "tenure":
                        st.info("Loyalty bonus: Provide a long-term retention offer or gift.")
                    elif top_feature == "TotalCharges":
                        st.info("Break total charges into easy installments or provide cashback.")
                    elif top_feature == "Contract":
                        st.info("Encourage switching to a long-term contract with benefits.")
                    elif top_feature == "TechSupport":
                        st.info("Offer free premium tech support for 3 months.")
                    elif top_feature == "InternetService":
                        st.info("Improve internet quality or provide speed upgrade trials.")
                    else:
                        st.info("Offer personalized support and a special loyalty discount.")

            else:
                st.success(f"Customer is not likely to churn.\nChurn Probability: **{probability:.2%}**")

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# =============== 📊 INSIGHTS PAGE ===============
elif page == "Model Insights":
    st.header("📊 Model Insights & Feature Importance")

    # Feature Importance Visualization
    st.subheader("Feature Importance")

    # Extract feature names (you can also load from joblib if available)
    feature_names = data.drop(columns=['Churn', 'customerID']).columns.tolist()

    # Check if model supports feature_importances_
    if hasattr(model, 'feature_importances_'):
        if len(model.feature_importances_) == len(feature_names):
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values(by="Importance", ascending=False)

            # Barplot of feature importance
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x='Importance', y='Feature', data=feature_importance, palette="viridis", ax=ax)
            ax.set_xlabel("Feature Importance Score")
            ax.set_ylabel("Features")
            ax.set_title("Top Features Influencing Churn")
            st.pyplot(fig)

            # Top 3 most important features
            top_factors = feature_importance.head(3)
            st.markdown("### 🔍 Top 3 Factors Contributing to Churn:")
            for i, row in top_factors.iterrows():
                st.write(f"**{row['Feature']}** – Importance: `{row['Importance']:.4f}`")

        else:
            st.error("Mismatch in feature importance length and feature names.")
            st.write(f"Feature importance length: {len(model.feature_importances_)}")
            st.write(f"Feature names length: {len(feature_names)}")
    else:
        st.warning("Feature importance is not available for this model.")

    # Show sample data
    st.subheader("📋 Sample Data from Dataset")
    st.dataframe(data.head(10))

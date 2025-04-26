import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the trained model, scaler, and label encoders
model = joblib.load("churn.pkl")
scaler_info = joblib.load("scaler1.pkl")
scaler = scaler_info["scaler"]
feature_names = scaler_info["numeric_cols"]
label_encoders = joblib.load("label_encoders.pkl")

# Load dataset for dashboard
df = pd.read_csv("Telecom Customers Churn.csv")

# Define feature names dynamically
# feature_names = scaler.feature_names_in_

# Streamlit App Configuration
st.set_page_config(page_title="Telecom Churn Prediction", layout="wide")

# Sidebar Navigation
st.sidebar.title("Churn Prediction App")
page = st.sidebar.radio("Go to:", ["Dashboard", "Predict Churn", "Model Insights"])

# =================== DASHBOARD PAGE ===================
if page == "Dashboard":
    st.title("📈 Customer Churn Analysis")

    # Ensure the 'Churn' column exists
    if 'Churn' not in df.columns:
        st.error("Error: 'Churn' column not found in dataset.")
        st.stop()

    # Convert 'Churn' column to binary (Yes → 1, No → 0)
    df["Churn"] = df["Churn"].map({'Yes': 1, 'No': 0})

    # Compute Churn Statistics
    total_customers = df.shape[0]
    churned_customers = df[df['Churn'] == 1].shape[0]
    churn_rate = round((churned_customers / total_customers) * 100, 2)

    # Display Key Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Churned Customers", f"{churned_customers:,}")
    col3.metric("Churn Rate", f"{churn_rate}%", delta=f"{churn_rate}% 🔻", delta_color="inverse")

    # Create columns for visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x=df['Churn'], palette=['green', 'red'], ax=ax)
        ax.set_title("Churn vs Active Customers")
        ax.set_xlabel("Churn (No = 0, Yes = 1)")
        ax.set_ylabel("Customer Count")
        st.pyplot(fig)

        # Churn Count by Different Factors
        st.subheader("Churn Count by Different Factors")

        # Set Seaborn style
        sns.set_style("whitegrid")

        # Create subplots (2x2 Grid)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        factor_columns = ["Contract", "InternetService", "TechSupport", "gender"]
        titles = [
            "Churn Count by Contract Type",
            "Churn Count by Internet Service",
            "Churn Count by Tech Support Availability",
            "Churn Count by Gender"
        ]

        # Iterate through factor columns & plot
        for i, col in enumerate(factor_columns):
            row, col_idx = divmod(i, 2)  # Compute subplot position
            sns.countplot(data=df, x=col, hue="Churn", ax=axes[row, col_idx], palette=["lightblue", "salmon"])
            axes[row, col_idx].set_title(titles[i])
            axes[row, col_idx].set_xlabel(col)
            axes[row, col_idx].set_ylabel("Count")
            axes[row, col_idx].tick_params(axis='x', rotation=30)

        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Correlation Heatmap")

        # Select numeric columns
        df_numeric = df.select_dtypes(include=[np.number])

        if df_numeric.shape[1] > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', ax=ax)
            ax.set_title("Feature Correlation Heatmap")
            st.pyplot(fig)
        else:
            st.warning("Not enough numerical data for correlation heatmap.")

# =================== PREDICTION PAGE ===================
elif page == "Predict Churn":
    st.title("🔍 Predict Customer Churn")

    # User Input Form
    with st.form("churn_form"):
        st.subheader("Enter Customer Details:")

        # Input fields for user data
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

        with col2:
            senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            monthly_bill = st.number_input("Monthly Charges", min_value=0.0, max_value=500.0, value=50.0)
            total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=600.0)

        # Submit button
        submit = st.form_submit_button("Predict Churn")

    # Process Prediction
    if submit:
        # Prepare input data
        input_data = pd.DataFrame([[gender, senior_citizen, partner, dependents, tenure, phone_service,
                                    multiple_lines, internet_service, online_security, online_backup,
                                    device_protection, tech_support, streaming_tv, streaming_movies,
                                    contract, paperless_billing, payment_method, monthly_bill, total_charges]],
                                  columns=["gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService",
                                            "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                                            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
                                            "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", "TotalCharges"])

        # Ensure all columns are present in input_data
        for col in input_data.columns:
            if col in label_encoders:
                input_data[col] = label_encoders[col].transform(input_data[col])
        
        # Scale numerical features
        input_data[feature_names] = scaler.transform(input_data[feature_names])

        # Ensure the column order matches what the model expects
        input_data = input_data[feature_names]

        # Scale numerical data
        # scaled_features = scaler.transform(input_data)

        # Make Prediction
        prediction = model.predict(input_data)
        churn_prob = model.predict_proba(input_data)[0][1] * 100  # Probability of churn

        # Display Result
        st.subheader("Prediction Result:")
        if prediction[0] == 1:
            st.error(f"This customer is **likely to churn** with a probability of **{churn_prob:.2f}%**")
        else:
            st.success(f"This customer is **not likely to churn** with a probability of **{(100-churn_prob):.2f}%**")

# =================== MODEL INSIGHTS PAGE ===================
elif page == "Model Insights":
    st.title("Model Performance & Insights")

    # Feature Importance Visualization
    st.subheader("Feature Importance")
    
    # Get feature names dynamically
    # feature_names = df.drop(columns=['Churn', 'customerID']).columns.tolist()

    if hasattr(model, 'feature_importances_'):
        if len(model.feature_importances_) == len(feature_names):  # Ensure lengths match
            feature_importance = pd.DataFrame({'Feature': feature_names,
                                               'Importance': model.feature_importances_})
            feature_importance = feature_importance.sort_values(by="Importance", ascending=False)

            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=feature_importance['Importance'], y=feature_importance['Feature'], palette="viridis", ax=ax)
            plt.xlabel("Feature Importance Score")
            plt.ylabel("Features")
            st.pyplot(fig)

             # Display the top 3 factors responsible for churn
            top_factors = feature_importance.head(3)
            st.write("The most influential factor in churn prediction are:")
            for idx, row in top_factors.iterrows():
                st.write(f"**{idx+1}. {row['Feature']}** - Importance: {row['Importance']:.4f}")

        else:
            st.error("Mismatch in feature importance length and feature names.")
            st.write(f"Feature importance length: {len(model.feature_importances_)}")
            st.write(f"Feature names length: {len(feature_names)}")
    else:
        st.warning("Feature importance is not available for this model.")

    # Show sample data
    st.subheader("Sample Data")
    st.dataframe(df.head(10))



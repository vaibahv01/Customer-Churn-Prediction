
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("customer_churn_svm_model.pkl")

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write("Predict whether a telecom customer is likely to leave the company.")

st.subheader("Enter Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (Months)", 0, 72, 12)
    phone = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox(
        "Multiple Lines", ["Yes", "No", "No phone service"]
    )

with col2:
    internet = st.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "No"]
    )
    security = st.selectbox(
        "Online Security", ["Yes", "No", "No internet service"]
    )
    backup = st.selectbox(
        "Online Backup", ["Yes", "No", "No internet service"]
    )
    device = st.selectbox(
        "Device Protection", ["Yes", "No", "No internet service"]
    )
    support = st.selectbox(
        "Tech Support", ["Yes", "No", "No internet service"]
    )
    tv = st.selectbox(
        "Streaming TV", ["Yes", "No", "No internet service"]
    )
    movies = st.selectbox(
        "Streaming Movies", ["Yes", "No", "No internet service"]
    )

with col3:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )
    monthly = st.number_input(
        "Monthly Charges", min_value=0.0, value=70.0
    )
    total = st.number_input(
        "Total Charges", min_value=0.0, value=840.0
    )

if st.button("Predict Customer Churn", type="primary"):

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    prediction = model.predict(customer)[0]
    probability = model.predict_proba(customer)[0][1]

    st.subheader("Prediction Result")

    st.metric(
        "Estimated Churn Probability",
        f"{probability * 100:.2f}%"
    )

    if prediction == 1:
        st.error("Customer is likely to churn.")
        st.write("Consider a retention offer or proactive customer support.")
    else:
        st.success("Customer is likely to stay.")
        st.write("Continue regular customer engagement.")


# -----------------------------------
# BULK CUSTOMER CHURN PREDICTION
# -----------------------------------

st.divider()

st.header("Bulk Customer Churn Prediction")

st.write("Upload a CSV file containing customer information.")

uploaded_file = st.file_uploader(
    "Upload Customer CSV",
    type=["csv"]
)

if uploaded_file is not None:

    bulk_data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Customer Data")
    st.dataframe(bulk_data, use_container_width=True)

    st.write("Total Customers Uploaded:", len(bulk_data))

    if st.button("Predict Bulk Churn"):

        customer_data = bulk_data.copy()

        # Remove columns not used during model training
        columns_to_remove = ["customerID", "Churn", "Tenure_Group"]

        customer_data = customer_data.drop(
            columns=columns_to_remove,
            errors="ignore"
        )

        # Numerical columns
        numerical_cols = [
            "SeniorCitizen",
            "tenure",
            "MonthlyCharges",
            "TotalCharges"
        ]

        # Convert numerical columns to numeric format
        for col in numerical_cols:
            customer_data[col] = pd.to_numeric(
                customer_data[col],
                errors="coerce"
            )

        # Handle missing numerical values
        customer_data["TotalCharges"] = customer_data["TotalCharges"].fillna(0)

        for col in ["SeniorCitizen", "tenure", "MonthlyCharges"]:
            customer_data[col] = customer_data[col].fillna(
                customer_data[col].median()
            )

        # Handle missing categorical values
        categorical_cols = [
            col for col in customer_data.columns
            if col not in numerical_cols
        ]

        customer_data[categorical_cols] = (
            customer_data[categorical_cols].fillna("Unknown")
        )

        # Predict
        predictions = model.predict(customer_data)

        probabilities = model.predict_proba(customer_data)[:, 1]

        bulk_data["Prediction"] = [
            "Churn" if p == 1 else "Not Churn"
            for p in predictions
        ]

        bulk_data["Churn Probability (%)"] = (
            probabilities * 100
        ).round(2)

        st.subheader("Bulk Prediction Results")

        st.dataframe(bulk_data, use_container_width=True)

        st.success(
            f"Successfully predicted {len(bulk_data)} customers!"
        )

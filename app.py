
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

    if st.button("Predict Bulk Churn"):

        required_columns = [
            "gender", "SeniorCitizen", "Partner", "Dependents",
            "tenure", "PhoneService", "MultipleLines",
            "InternetService", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaperlessBilling",
            "PaymentMethod", "MonthlyCharges", "TotalCharges"
        ]

        missing_columns = [
            col for col in required_columns
            if col not in bulk_data.columns
        ]

        if missing_columns:
            st.error(f"Missing columns: {missing_columns}")

        else:
            customer_data = bulk_data[required_columns].copy()

            predictions = model.predict(customer_data)
            probabilities = model.predict_proba(customer_data)[:, 1]

            bulk_data["Churn_Prediction"] = [
                "Yes" if p == 1 else "No"
                for p in predictions
            ]

            bulk_data["Churn_Probability"] = (
                probabilities * 100
            ).round(2)

            st.subheader("Prediction Results")
            st.dataframe(bulk_data)

            csv = bulk_data.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Prediction Results",
                data=csv,
                file_name="customer_churn_predictions.csv",
                mime="text/csv"
            )

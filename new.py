import pandas as pd
import streamlit as st
import joblib
import base64

def set_background(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("UPI.webp")

st.markdown("<h1 style='text-align: center; color: red;'> UPI Fraud Detection System</h1>", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load('fraud_model.joblib')

model = load_model()

with st.form("fraud_form"):
    st.markdown("### Enter Transaction Details Below")
    
    
    upi_id = st.text_input("Enter UPI ID", placeholder="e.g., user@upi")

    # List of valid UPI domains
    valid_domains = [
        "@sbi", "@axisbank", "@hdfcbank", "@icici", "@pnb", "@indus",
        "@barodampay", "@canarabank", "@unionbank", "@boi",
        "@paytm", "@phonepe", "@okaxis"
    ]

    if upi_id:
        if not any(upi_id.endswith(domain) for domain in valid_domains):
            st.warning("⚠️ UPI ID  not recognized! Possible Fraud.")
        else:
            st.success("✅ UPI ID  is recognized.")

    utr = st.text_input("Enter UTR (Unique Transaction Reference)", placeholder="e.g., 1234UPI5678XYZ")
    step = st.number_input("Step (hour in simulation)", value=1)
    amount = st.number_input("Transaction amount", value=1000.0)
    oldbalanceOrg = st.number_input("Sender's balance before transaction", value=5000.0)
    newbalanceOrig = st.number_input("Sender's balance after transaction", value=4000.0)
    oldbalanceDest = st.number_input("Receiver's balance before transaction", value=2000.0)
    newbalanceDest = st.number_input("Receiver's balance after transaction", value=3000.0)
    type_input = st.selectbox("Transaction type", ['TRANSFER', 'PAYMENT','CASH_OUT','CASH_IN'])

    submitted = st.form_submit_button("Predict")

hour = step % 24
is_night = 1 if hour < 6 else 0
amount_ratio = amount / (oldbalanceOrg + 1)
sender_balance_change = oldbalanceOrg - newbalanceOrig
receiver_balance_change = newbalanceDest - oldbalanceDest
orig_balance_zero = 1 if oldbalanceOrg == 0 else 0
dest_balance_zero = 1 if oldbalanceDest == 0 else 0
type_TRANSFER = 1 if type_input == "TRANSFER" else 0

input_data = pd.DataFrame([{
    'step': step,
    'amount': amount,
    'oldbalanceOrg': oldbalanceOrg,
    'newbalanceOrig': newbalanceOrig,
    'oldbalanceDest': oldbalanceDest,
    'newbalanceDest': newbalanceDest,
    'hour': hour,
    'is_night': is_night,
    'amount_ratio': amount_ratio,
    'sender_balance_change': sender_balance_change,
    'receiver_balance_change': receiver_balance_change,
    'orig_balance_zero': orig_balance_zero,
    'dest_balance_zero': dest_balance_zero,
    'type_TRANSFER': type_TRANSFER
}])

if st.button("🔍 Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Transaction is Legitimate.")

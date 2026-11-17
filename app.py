import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Industrial Equipment Health Monitor",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 AI-Based Industrial Equipment Health Monitoring")
st.caption("Machine Learning Based Predictive Maintenance System")

@st.cache_resource
def train_model():
    np.random.seed(42)
    samples = 5000

    temperature = np.random.uniform(35, 115, samples)
    usage = np.random.randint(1, 4, samples)
    hours = np.random.uniform(1, 24, samples)
    total_hours = np.random.uniform(100, 20000, samples)
    maintenance = np.random.uniform(0, 365, samples)

    risk = (
        0.30 * np.clip((temperature - 60) / 50, 0, 1)
        + 0.18 * ((usage - 1) / 2)
        + 0.18 * (hours / 24)
        + 0.20 * (total_hours / 20000)
        + 0.14 * (maintenance / 365)
    )
    risk += np.random.normal(0, 0.055, samples)

    labels = np.where(risk >= 0.67, 2, np.where(risk >= 0.43, 1, 0))

    X = pd.DataFrame({
        "temperature": temperature,
        "usage": usage,
        "hours": hours,
        "total_hours": total_hours,
        "maintenance": maintenance
    })

    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )

    model = RandomForestClassifier(
        n_estimators=140, max_depth=10, random_state=42
    )
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy

model, accuracy = train_model()

st.sidebar.header("Machine Parameters")
machine_name = st.sidebar.text_input("Machine Name", "Industrial Motor 01")
temperature = st.sidebar.slider("Temperature (°C)", 20, 130, 75)
usage_label = st.sidebar.selectbox("Machine Usage", ["Light", "Medium", "Heavy"])
hours = st.sidebar.slider("Operating Hours Per Day", 1, 24, 12)
total_hours = st.sidebar.number_input("Total Operating Hours", min_value=0, value=5000)
maintenance = st.sidebar.number_input("Days Since Last Maintenance", min_value=0, value=60)

usage_map = {"Light": 1, "Medium": 2, "Heavy": 3}
input_data = pd.DataFrame([{
    "temperature": temperature,
    "usage": usage_map[usage_label],
    "hours": hours,
    "total_hours": total_hours,
    "maintenance": maintenance
}])

prediction = int(model.predict(input_data)[0])
probabilities = model.predict_proba(input_data)[0]
risk_probability = float(probabilities[1] * 0.5 + probabilities[2])
health_score = round((1 - risk_probability) * 100)

recommendations = {
    0: "Machine is operating normally. Continue regular monitoring.",
    1: "Warning! Schedule machine inspection and maintenance.",
    2: "Critical condition! Arrange immediate inspection and assess whether operation should be stopped safely."
}

st.subheader(f"Machine: {machine_name}")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Machine Health", f"{health_score}%")
col2.metric("Failure Risk", f"{risk_probability * 100:.1f}%")
col3.metric("Temperature", f"{temperature} °C")
col4.metric("Operating Hours", f"{hours} hrs/day")

st.divider()
left, right = st.columns(2)

with left:
    st.subheader("Machine Condition")
    if prediction == 0:
        st.success("NORMAL OPERATION")
    elif prediction == 1:
        st.warning("WARNING: MAINTENANCE REQUIRED")
    else:
        st.error("CRITICAL: HIGH FAILURE RISK")
    st.info(recommendations[prediction])

with right:
    st.subheader("Failure Risk Analysis")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_probability * 100,
        number={"suffix": "%"},
        gauge={
            "axis": {"range": [0, 100]},
            "steps": [
                {"range": [0, 35], "color": "#d1fae5"},
                {"range": [35, 65], "color": "#fef3c7"},
                {"range": [65, 100], "color": "#fee2e2"}
            ]
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Machine Parameter Summary")
summary = pd.DataFrame({
    "Parameter": [
        "Temperature", "Usage Intensity", "Operating Hours Per Day",
        "Total Operating Hours", "Days Since Maintenance"
    ],
    "Value": [
        f"{temperature} °C", usage_label, f"{hours} hours",
        f"{total_hours} hours", f"{maintenance} days"
    ]
})
st.dataframe(summary, use_container_width=True, hide_index=True)

st.subheader("Machine Learning Information")
st.write("**Algorithm:** Random Forest Classifier")
st.write(f"**Demo test accuracy:** {accuracy * 100:.2f}%")
st.caption(
    "The model is trained on synthetic demonstration data. "
    "Its test accuracy is not evidence of real industrial performance."
)
st.warning(
    "Academic prototype only. Real industrial deployment requires actual "
    "machine failure records, validation, and safety-reviewed maintenance thresholds."
)

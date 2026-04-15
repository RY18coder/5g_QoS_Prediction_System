# ================= AUTO INSTALL (FIRST RUN ONLY) =================
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import streamlit as st
except:
    install("streamlit")
    import streamlit as st

try:
    import pandas as pd
except:
    install("pandas")
    import pandas as pd

try:
    import numpy as np
except:
    install("numpy")
    import numpy as np

try:
    import joblib
except:
    install("joblib")
    import joblib

try:
    import matplotlib.pyplot as plt
except:
    install("matplotlib")
    import matplotlib.pyplot as plt


# ================= LOAD MODEL =================
model = joblib.load("rf_model.pkl")

# ================= PAGE CONFIG =================
st.set_page_config(page_title="5G QoS AI System", layout="wide")

# ================= TITLE =================
st.title("📡 Intelligent 5G QoS Prediction System")
st.write("AI-powered QoS prediction, analysis & optimization")

# ================= SIDEBAR INPUT =================
st.sidebar.header("📥 Input Network Parameters")

speed = st.sidebar.slider("Speed (km/h)", 0, 120, 40)
snr = st.sidebar.slider("SNR", 0, 50, 25)
mcs = st.sidebar.slider("MCS", 0, 30, 18)
rbs = st.sidebar.slider("Resource Blocks", 0, 100, 50)

st.sidebar.markdown("---")
st.sidebar.markdown("### Project Team")

st.sidebar.markdown("""
- Udbhav Mathur - 21111502823
- Rohit Yadav - 21711502823
- Siddhant Arya - 21811502823
- Nandini Tiwari - 22111502823
""")

# ================= SLICE LOGIC =================
def assign_slice(speed, snr, mcs):
    if speed > 60 and snr > 20:
        return "URLLC"
    elif mcs > 15:
        return "eMBB"
    else:
        return "mMTC"

slice_type = assign_slice(speed, snr, mcs)

# ================= CREATE INPUT =================
input_data = pd.DataFrame([{
    'speed_kmh': speed,
    'pcell_snr_max': snr,
    'scell_snr_max': snr,
    'pcell_downlink_num_rbs': rbs,
    'scell_downlink_num_rbs': rbs,
    'pcell_downlink_average_mcs': mcs,
    'scell_downlink_average_mcs': mcs
}])

# Fill missing columns
for col in model.feature_names_in_:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[model.feature_names_in_]

# ================= PREDICTION =================
qos = model.predict(input_data)[0]

# Keep track of observed QoS
if "qos_min" not in st.session_state:
    st.session_state.qos_min = qos
    st.session_state.qos_max = qos

st.session_state.qos_min = min(st.session_state.qos_min, qos)
st.session_state.qos_max = max(st.session_state.qos_max, qos)

# Normalize
range_qos = st.session_state.qos_max - st.session_state.qos_min

if range_qos == 0:
    normalized = 0.5
else:
    normalized = (qos - st.session_state.qos_min) / range_qos

# Categorize
if normalized > 0.7:
    category = "HIGH"
elif normalized > 0.4:
    category = "MEDIUM"
else:
    category = "LOW"

# ================= MODEL ACCURACY =================
# (Use your known value)
accuracy = 0.946  # your Random Forest R2

# ================= DASHBOARD =================
col1, col2, col3 , col4 = st.columns(4)

col1.metric("📡 Slice Type", slice_type)
col2.metric("📊 Predicted QoS", f"{qos:,.0f}")
col3.metric("⚡ QoS Category", category)
col4.metric("🎯 Model Accuracy (R²)", accuracy)

# ================= ADVISOR =================
st.subheader("💡 AI Recommendations")

if category == "LOW":
    if snr < 20:
        st.warning("Increase Signal Strength (SNR)")
    if rbs < 50:
        st.warning("Allocate more Resource Blocks")
    if mcs < 15:
        st.warning("Improve Modulation (MCS)")
else:
    st.success("Network conditions are optimal")

# ================= GRAPH =================
st.subheader("📈 QoS Sensitivity Analysis")

feature = st.selectbox("Select Parameter", ["SNR", "Resource Blocks", "MCS"])

if feature == "SNR":
    values = np.arange(5, 50, 5)
    feature_col = 'pcell_snr_max'
elif feature == "Resource Blocks":
    values = np.arange(10, 100, 10)
    feature_col = 'pcell_downlink_num_rbs'
else:
    values = np.arange(5, 30, 2)
    feature_col = 'pcell_downlink_average_mcs'

qos_values = []

for v in values:
    temp = input_data.copy()
    temp[feature_col] = v
    
    pred = model.predict(temp)[0]
    qos_values.append(pred)

fig, ax = plt.subplots()
ax.plot(values, qos_values)
ax.set_xlabel(feature)
ax.set_ylabel("QoS")
ax.set_title(f"Effect of {feature} on QoS")

st.pyplot(fig)

# ================= WHAT-IF SIMULATOR (ADVANCED UI) =================
st.subheader("🔍 What-If QoS Improvement Simulator")

st.markdown("Adjust improvements and see how QoS changes in real-time")

# --- Improvement sliders ---
colA, colB, colC = st.columns(3)

snr_boost = colA.slider("Increase SNR", 0, 20, 5)
rbs_boost = colB.slider("Increase RBs", 0, 50, 10)
mcs_boost = colC.slider("Increase MCS", 0, 10, 2)

# --- Create improved input ---
improved_input = input_data.copy()
improved_input['pcell_snr_max'] += snr_boost
improved_input['scell_snr_max'] += snr_boost
improved_input['pcell_downlink_num_rbs'] += rbs_boost
improved_input['scell_downlink_num_rbs'] += rbs_boost
improved_input['pcell_downlink_average_mcs'] += mcs_boost
improved_input['scell_downlink_average_mcs'] += mcs_boost

# --- Predictions ---
current_qos = qos
improved_qos = model.predict(improved_input)[0]

# --- Improvement calculation ---
improvement = ((improved_qos - current_qos) / current_qos) * 100

# --- Display comparison ---
st.markdown("### 📊 Before vs After Comparison")

col1, col2, col3 = st.columns(3)

col1.metric("Current QoS", f"{current_qos:,.0f}")
col2.metric("Improved QoS", f"{improved_qos:,.0f}")
col3.metric("Improvement %", f"{improvement:.2f}%", delta=f"{improvement:.2f}%")

# --- Visual Indicator ---
st.markdown("### 📈 QoS Improvement Visualization")

import matplotlib.pyplot as plt

labels = ['Current QoS', 'Improved QoS']
values = [current_qos, improved_qos]

fig, ax = plt.subplots()
ax.bar(labels, values)
ax.set_ylabel("QoS")
ax.set_title("QoS Improvement Comparison")

st.pyplot(fig)

# --- Smart Insight ---
st.markdown("### 💡 Insight")

if improvement > 20:
    st.success("Significant improvement achieved 🚀")
elif improvement > 5:
    st.info("Moderate improvement observed 👍")
else:
    st.warning("Minimal improvement — try increasing parameters more")
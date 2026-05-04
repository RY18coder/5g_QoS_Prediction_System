# ================= IMPORTS =================
import subprocess
import sys
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from utils.optimizer import optimize_network

# ================= LOAD MODEL =================
model = joblib.load("rf_model.pkl")
feature_columns = list(model.feature_names_in_)

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
- Udbhav Mathur  
- Rohit Yadav  
- Siddhant Arya  
- Nandini Tiwari  
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
for col in feature_columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[feature_columns]

# ================= PREDICTION =================
qos = model.predict(input_data)[0]

# ================= NORMALIZATION =================
if "qos_min" not in st.session_state:
    st.session_state.qos_min = qos
    st.session_state.qos_max = qos

st.session_state.qos_min = min(st.session_state.qos_min, qos)
st.session_state.qos_max = max(st.session_state.qos_max, qos)

range_qos = st.session_state.qos_max - st.session_state.qos_min

normalized = 0.5 if range_qos == 0 else (qos - st.session_state.qos_min) / range_qos

if normalized > 0.7:
    category = "HIGH"
elif normalized > 0.4:
    category = "MEDIUM"
else:
    category = "LOW"

# ================= MODEL ACCURACY =================
accuracy = 0.946

# ================= DASHBOARD =================
col1, col2, col3, col4 = st.columns(4)

col1.metric("📡 Slice Type", slice_type)
col2.metric("📊 Predicted QoS", f"{qos:,.0f}")
col3.metric("⚡ QoS Category", category)
col4.metric("🎯 Model Accuracy (R²)", accuracy)

# ================= AI OPTIMIZATION ENGINE =================
st.subheader("🚀 AI Network Optimization (Cost-Aware)")

optimized = optimize_network(
    model=model,
    base_input=input_data.iloc[0].to_dict(),
    feature_order=feature_columns
)

# Cost calculation
current_cost = 2*rbs + 1.5*mcs
optimized_cost = 2*optimized['rbs'] + 1.5*optimized['mcs']

colA, colB, colC = st.columns(3)

colA.metric("Optimized QoS", f"{optimized['qos']:,.0f}")
colB.metric("QoS Gain", f"{((optimized['qos']-qos)/qos)*100:.2f}%")
colC.metric("Cost Change", f"{optimized_cost-current_cost:.2f}")

st.markdown("### 🔧 Optimal Configuration")
st.write(f"• Resource Blocks: {rbs} → {optimized['rbs']}")
st.write(f"• MCS: {mcs} → {optimized['mcs']}")
st.write(f"• SNR: {snr} → {optimized['snr']}")
st.write(f"• Recommended Slice: **{optimized['slice']}**")

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
    qos_values.append(model.predict(temp)[0])

fig, ax = plt.subplots()
ax.plot(values, qos_values)
ax.set_xlabel(feature)
ax.set_ylabel("QoS")
ax.set_title(f"Effect of {feature} on QoS")

st.pyplot(fig)

# ================= WHAT-IF SIMULATOR =================
st.subheader("🔍 What-If QoS Improvement Simulator")

colA, colB, colC = st.columns(3)

snr_boost = colA.slider("Increase SNR", 0, 20, 5)
rbs_boost = colB.slider("Increase RBs", 0, 50, 10)
mcs_boost = colC.slider("Increase MCS", 0, 10, 2)

improved_input = input_data.copy()
improved_input['pcell_snr_max'] += snr_boost
improved_input['scell_snr_max'] += snr_boost
improved_input['pcell_downlink_num_rbs'] += rbs_boost
improved_input['scell_downlink_num_rbs'] += rbs_boost
improved_input['pcell_downlink_average_mcs'] += mcs_boost
improved_input['scell_downlink_average_mcs'] += mcs_boost

improved_qos = model.predict(improved_input)[0]
improvement = ((improved_qos - qos) / qos) * 100

col1, col2, col3 = st.columns(3)

col1.metric("Current QoS", f"{qos:,.0f}")
col2.metric("Improved QoS", f"{improved_qos:,.0f}")
col3.metric("Improvement %", f"{improvement:.2f}%")

fig, ax = plt.subplots()
ax.bar(['Current', 'Improved'], [qos, improved_qos])
st.pyplot(fig)

if improvement > 20:
    st.success("Significant improvement achieved 🚀")
elif improvement > 5:
    st.info("Moderate improvement 👍")
else:
    st.warning("Try increasing parameters more")

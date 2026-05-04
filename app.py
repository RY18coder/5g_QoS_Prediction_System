# ================= IMPORTS =================
import subprocess
import sys
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ================= LOAD MODEL =================
model = joblib.load("rf_model.pkl")
feature_columns = list(model.feature_names_in_)

# ================= PAGE CONFIG =================
st.set_page_config(page_title="5G QoS AI System", layout="wide")

# ================= TITLE =================
st.title("📡 Intelligent 5G QoS Prediction System")
st.write("AI-powered QoS prediction, analysis & optimization")

# ================= SIDEBAR =================
st.sidebar.header("📥 Input Network Parameters")

slice_type = st.sidebar.selectbox(
    "Select 5G Slice Type",
    ["eMBB", "URLLC", "mMTC"]
)

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

for col in feature_columns:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[feature_columns]

# ================= PREDICTION =================
qos = model.predict(input_data)[0]

# ================= QoS CATEGORY =================
if qos > 8e7:
    category = "HIGH"
elif qos > 4e7:
    category = "MEDIUM"
else:
    category = "LOW"

# ================= DASHBOARD =================
col1, col2, col3, col4 = st.columns(4)

col1.metric("📡 Slice Type", slice_type)
col2.metric("📊 Predicted QoS", f"{qos:,.0f}")
col3.metric("⚡ QoS Category", category)
col4.metric("🎯 Model Accuracy (R²)", 0.946)

# ================= REAL OPTIMIZATION ENGINE =================
st.subheader("🚀 Slice-Aware Network Optimization")

# Slice constraints
if slice_type == "eMBB":
    bounds = [(50, 100), (15, 30), (15, 40)]
elif slice_type == "URLLC":
    bounds = [(20, 60), (10, 25), (25, 40)]
else:
    bounds = [(10, 50), (5, 15), (10, 30)]

def objective(x):
    rbs_opt, mcs_opt, snr_opt = x

    temp = input_data.copy()
    temp['pcell_downlink_num_rbs'] = rbs_opt
    temp['scell_downlink_num_rbs'] = rbs_opt
    temp['pcell_downlink_average_mcs'] = mcs_opt
    temp['scell_downlink_average_mcs'] = mcs_opt
    temp['pcell_snr_max'] = snr_opt
    temp['scell_snr_max'] = snr_opt

    pred_qos = model.predict(temp)[0]

    # resource penalty (realistic tradeoff)
    penalty = (rbs_opt/100) + (mcs_opt/30)

    return -(pred_qos - 0.1 * penalty)

x0 = [rbs, mcs, snr]

result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
opt_rbs, opt_mcs, opt_snr = result.x

# Apply optimized values
opt_input = input_data.copy()
opt_input['pcell_downlink_num_rbs'] = opt_rbs
opt_input['scell_downlink_num_rbs'] = opt_rbs
opt_input['pcell_downlink_average_mcs'] = opt_mcs
opt_input['scell_downlink_average_mcs'] = opt_mcs
opt_input['pcell_snr_max'] = opt_snr
opt_input['scell_snr_max'] = opt_snr

optimized_qos = model.predict(opt_input)[0]
improvement = ((optimized_qos - qos) / qos) * 100

# ================= RESULTS =================
st.markdown("### 📊 Optimization Results")

c1, c2, c3 = st.columns(3)
c1.metric("Current QoS", f"{qos:,.0f}")
c2.metric("Optimized QoS", f"{optimized_qos:,.0f}")
c3.metric("Improvement", f"{improvement:.2f}%")

# ================= PARAMETER CHANGES =================
st.markdown("### 🔧 Recommended Network Changes")

st.write(f"Resource Blocks: {rbs} → {round(opt_rbs,2)}")
st.write(f"MCS: {mcs} → {round(opt_mcs,2)}")
st.write(f"SNR: {snr} → {round(opt_snr,2)}")

# ================= REAL COST / IMPACT ANALYSIS =================
st.markdown("### 💰 Network Impact Analysis")

if opt_rbs > rbs:
    st.warning("More bandwidth allocation required → Higher operational cost")
else:
    st.success("Efficient bandwidth usage → Cost saving")

if opt_snr > snr:
    st.info("Requires better signal quality → Tower/infrastructure tuning needed")

if opt_mcs > mcs:
    st.info("Higher modulation → Better channel conditions required")

# ================= ADVISOR =================
st.subheader("💡 AI Recommendations")

if category == "LOW":
    if snr < 20:
        st.warning("Improve signal strength")
    if rbs < 50:
        st.warning("Increase bandwidth allocation")
    if mcs < 15:
        st.warning("Improve modulation scheme")
else:
    st.success("Network performing well")

# ================= GRAPH =================
st.subheader("📈 QoS Sensitivity Analysis")

feature = st.selectbox("Select Parameter", ["SNR", "Resource Blocks", "MCS"])

if feature == "SNR":
    values = np.arange(5, 50, 5)
    col = 'pcell_snr_max'
elif feature == "Resource Blocks":
    values = np.arange(10, 100, 10)
    col = 'pcell_downlink_num_rbs'
else:
    values = np.arange(5, 30, 2)
    col = 'pcell_downlink_average_mcs'

qos_vals = []

for v in values:
    temp = input_data.copy()
    temp[col] = v
    qos_vals.append(model.predict(temp)[0])

fig, ax = plt.subplots()
ax.plot(values, qos_vals)
ax.set_xlabel(feature)
ax.set_ylabel("QoS")
st.pyplot(fig)

# ================= WHAT-IF =================
st.subheader("🔍 What-If Simulator")

colA, colB, colC = st.columns(3)

snr_boost = colA.slider("Increase SNR", 0, 20, 5)
rbs_boost = colB.slider("Increase RBs", 0, 50, 10)
mcs_boost = colC.slider("Increase MCS", 0, 10, 2)

temp = input_data.copy()
temp['pcell_snr_max'] += snr_boost
temp['scell_snr_max'] += snr_boost
temp['pcell_downlink_num_rbs'] += rbs_boost
temp['scell_downlink_num_rbs'] += rbs_boost
temp['pcell_downlink_average_mcs'] += mcs_boost
temp['scell_downlink_average_mcs'] += mcs_boost

new_qos = model.predict(temp)[0]

st.metric("Improved QoS", f"{new_qos:,.0f}")

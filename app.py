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
st.title("📡 5G QoS Decision Intelligence System")
st.write("AI-powered QoS prediction, SLA validation & network optimization")

# ================= SIDEBAR =================
st.sidebar.header("📥 Network Inputs")

slice_type = st.sidebar.selectbox("5G Slice Type", ["eMBB", "URLLC", "mMTC"])

target_qos = st.sidebar.number_input(
    "🎯 Target QoS (SLA)",
    min_value=10000000,
    max_value=200000000,
    value=80000000,
    step=1000000
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

# ================= INPUT DATA =================
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

# ================= CURRENT QoS =================
qos = model.predict(input_data)[0]

# ================= CATEGORY =================
if qos > 8e7:
    category = "HIGH"
elif qos > 4e7:
    category = "MEDIUM"
else:
    category = "LOW"

# ================= DASHBOARD =================
c1, c2, c3, c4 = st.columns(4)

c1.metric("Slice", slice_type)
c2.metric("Current QoS", f"{qos:,.0f}")
c3.metric("QoS Category", category)
c4.metric("Model Accuracy", "94.6%")

# ================= SLA OPTIMIZATION =================
st.subheader("🚀 SLA-Based Network Optimization")

if slice_type == "eMBB":
    bounds = [(50, 100), (15, 30), (15, 40)]
elif slice_type == "URLLC":
    bounds = [(20, 60), (10, 25), (25, 40)]
else:
    bounds = [(10, 50), (5, 15), (10, 30)]

def sla_objective(x):
    rbs_o, mcs_o, snr_o = map(int, x)

    temp = input_data.copy()
    temp['pcell_downlink_num_rbs'] = rbs_o
    temp['scell_downlink_num_rbs'] = rbs_o
    temp['pcell_downlink_average_mcs'] = mcs_o
    temp['scell_downlink_average_mcs'] = mcs_o
    temp['pcell_snr_max'] = snr_o
    temp['scell_snr_max'] = snr_o

    pred = model.predict(temp)[0]

    penalty = (rbs_o/100) + (mcs_o/30)

    return abs(pred - target_qos) + 1e7 * penalty

result = minimize(sla_objective, [rbs, mcs, snr], bounds=bounds, method='L-BFGS-B')

opt_rbs = int(round(result.x[0]))
opt_mcs = int(round(result.x[1]))
opt_snr = int(round(result.x[2]))

# recompute QoS
temp = input_data.copy()
temp['pcell_downlink_num_rbs'] = opt_rbs
temp['scell_downlink_num_rbs'] = opt_rbs
temp['pcell_downlink_average_mcs'] = opt_mcs
temp['scell_downlink_average_mcs'] = opt_mcs
temp['pcell_snr_max'] = opt_snr
temp['scell_snr_max'] = opt_snr

optimized_qos = model.predict(temp)[0]

# ================= FEASIBILITY =================
st.markdown("### 🧠 SLA Feasibility")

if optimized_qos >= target_qos * 0.95:
    st.success("✅ SLA ACHIEVABLE")
else:
    st.error("❌ SLA NOT ACHIEVABLE")

    if slice_type == "mMTC":
        st.write("- mMTC not designed for high throughput")
    if snr < 15:
        st.write("- Poor signal quality")
    if rbs < 30:
        st.write("- Low bandwidth allocation")

# ================= RESULTS =================
st.markdown("### 📊 Optimization Results")

r1, r2, r3 = st.columns(3)
r1.metric("Target QoS", f"{target_qos:,.0f}")
r2.metric("Achieved QoS", f"{optimized_qos:,.0f}")
r3.metric("Gap", f"{target_qos - optimized_qos:,.0f}")

# ================= AI DECISION ENGINE =================
st.markdown("### 💡 AI Decision Engine")

delta_rbs = opt_rbs - rbs
delta_mcs = opt_mcs - mcs
delta_snr = opt_snr - snr

if optimized_qos >= target_qos:
    st.success("Optimal configuration found")

if delta_rbs > 0:
    st.write(f"➡ Increase RBs by {delta_rbs}")
elif delta_rbs < 0:
    st.write(f"➡ Reduce RBs by {abs(delta_rbs)} (save cost)")

if delta_mcs > 0:
    st.write(f"➡ Increase MCS to {opt_mcs}")
elif delta_mcs < 0:
    st.write("➡ Reduce MCS for stability")

if delta_snr > 0:
    st.write(f"➡ Improve SNR by {delta_snr}")

# ================= IMPACT =================
st.markdown("### 💰 Operational Impact")

if delta_rbs > 0:
    st.warning("Higher spectrum usage → Higher cost")
elif delta_rbs < 0:
    st.success("Spectrum saving → Cost reduction")

if delta_snr > 0:
    st.info("Infra upgrade required")

# ================= GRAPH =================
st.subheader("📈 QoS Sensitivity")

feature = st.selectbox("Parameter", ["SNR", "RBs", "MCS"])

if feature == "SNR":
    vals = np.arange(5, 50, 5)
    col = 'pcell_snr_max'
elif feature == "RBs":
    vals = np.arange(10, 100, 10)
    col = 'pcell_downlink_num_rbs'
else:
    vals = np.arange(5, 30, 2)
    col = 'pcell_downlink_average_mcs'

qos_vals = []
for v in vals:
    temp = input_data.copy()
    temp[col] = v
    qos_vals.append(model.predict(temp)[0])

fig, ax = plt.subplots()
ax.plot(vals, qos_vals)
st.pyplot(fig)

# ================= WHAT-IF =================
st.subheader("🔍 What-If Simulator")

a, b, c = st.columns(3)

snr_b = a.slider("SNR Boost", 0, 20, 5)
rbs_b = b.slider("RB Boost", 0, 50, 10)
mcs_b = c.slider("MCS Boost", 0, 10, 2)

temp = input_data.copy()
temp['pcell_snr_max'] += snr_b
temp['pcell_downlink_num_rbs'] += rbs_b
temp['pcell_downlink_average_mcs'] += mcs_b

new_qos = model.predict(temp)[0]

st.metric("New QoS", f"{new_qos:,.0f}")

fig, ax = plt.subplots()
ax.bar(["Current", "New"], [qos, new_qos])
st.pyplot(fig)

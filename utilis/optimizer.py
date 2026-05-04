import numpy as np
from scipy.optimize import minimize

# ---- Cost Function (Telecom Perspective) ----
def compute_cost(rbs, mcs, snr):
    # You can tune weights later
    return (2.0 * rbs) + (1.5 * mcs) - (0.5 * snr)

# ---- Slice Selection Logic ----
def get_slice_type(rbs, mcs, snr):
    if rbs > 70 and mcs > 20:
        return "eMBB"
    elif snr > 25:
        return "URLLC"
    else:
        return "mMTC"

# ---- Optimization Objective ----
def objective(x, model, base_input, feature_order, cost_weight=0.02):
    rbs, mcs, snr = x

    temp = base_input.copy()

    temp['total_rbs'] = rbs
    temp['pcell_downlink_average_mcs'] = mcs
    temp['total_snr'] = snr

    # Ensure correct order
    input_vector = np.array([temp[f] for f in feature_order]).reshape(1, -1)

    pred_qos = model.predict(input_vector)[0]

    cost = compute_cost(rbs, mcs, snr)

    # Maximize QoS, minimize cost → convert to minimization
    return -(pred_qos - cost_weight * cost)


# ---- MAIN OPTIMIZATION FUNCTION ----
def optimize_network(model, base_input, feature_order):

    # Initial guess (current state)
    x0 = [
        base_input.get('total_rbs', 50),
        base_input.get('pcell_downlink_average_mcs', 15),
        base_input.get('total_snr', 20)
    ]

    # Bounds (realistic telecom limits)
    bounds = [
        (10, 100),   # RBs
        (5, 30),     # MCS
        (5, 40)      # SNR
    ]

    result = minimize(
        objective,
        x0,
        args=(model, base_input, feature_order),
        bounds=bounds,
        method='L-BFGS-B'
    )

    optimal_rbs, optimal_mcs, optimal_snr = result.x

    # Final QoS
    temp = base_input.copy()
    temp['total_rbs'] = optimal_rbs
    temp['pcell_downlink_average_mcs'] = optimal_mcs
    temp['total_snr'] = optimal_snr

    input_vector = np.array([temp[f] for f in feature_order]).reshape(1, -1)
    final_qos = model.predict(input_vector)[0]

    slice_type = get_slice_type(optimal_rbs, optimal_mcs, optimal_snr)

    return {
        "qos": final_qos,
        "rbs": round(optimal_rbs, 2),
        "mcs": round(optimal_mcs, 2),
        "snr": round(optimal_snr, 2),
        "slice": slice_type
    }

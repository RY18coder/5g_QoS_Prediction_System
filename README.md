📡 Intelligent QoS Prediction for 5G Network Slices using Machine Learning

🚀 Overview

This project presents an AI-powered system for predicting and optimizing Quality of Service (QoS) in 5G network slices. It leverages machine learning models to forecast network performance metrics and provide actionable recommendations for improving QoS in real-time.
The system is deployed as an interactive web application using Streamlit, enabling users to simulate network conditions and analyze their impact on QoS.

🎯 Problem Statement
Traditional network management systems are reactive and fail to detect subtle performance degradations (e.g., jitter, latency spikes) in dynamic 5G environments.
This project addresses:
Lack of predictive QoS monitoring
Inability to detect “grey failures”
No intelligent decision-support system

💡 Solution
We propose a predictive + advisory system that:
Predicts QoS using machine learning
Classifies network performance (High / Medium / Low)
Identifies 5G slice types (eMBB, URLLC, mMTC)
Provides optimization recommendations
Enables real-time simulation via web app

🧠 Machine Learning Models Used
Linear Regression
Random Forest (Best performing)
Gradient Boosting
📊 Model Performance

Model R² Score
Linear Regression: 0.7587
Random Forest: 0.9460
Gradient Boosting: 0.9398
Random Forest achieved the highest accuracy and was selected for deployment.

⚙️ Features
🔮 QoS Prediction
Predicts network QoS based on input parameters:
Speed
SNR (Signal-to-Noise Ratio)
MCS (Modulation Coding Scheme)
Resource Blocks
📡 5G Slice Classification
Automatically identifies slice type:
eMBB → High throughput
URLLC → Low latency
mMTC → IoT devices
⚡ QoS Categorization
HIGH → Optimal performance
MEDIUM → Acceptable performance
LOW → Needs improvement
💡 AI-Based Recommendations
Provides actionable suggestions such as:
Increase signal strength (SNR)
Allocate more resource blocks
Improve modulation scheme
📈 Real-Time Simulation
Interactive dashboard allows:
Parameter tuning via sliders
Live QoS updates
Sensitivity analysis graphs
🔍 What-If Simulator
Simulates improvements:
Compare current vs improved QoS
Visualize performance gain
Decision-support system

🌐 Web Application
The system is deployed using:
Streamlit
Run Locally
Bash
pip install -r requirements.txt
python -m streamlit run app.py
📁 Project Structure

5g_qos_prediction_system/
│
├── app.py                 # Streamlit web app
├── rf_model.pkl           # Trained ML model
├── requirements.txt       # Dependencies
├── train.csv              # Dataset (optional)
└── README.md              # Project documentation

🛠️ Tech Stack
Python
Pandas
NumPy
scikit-learn
Matplotlib
Joblib
Streamlit

📊 Key Insights
QoS is highly influenced by:
Signal strength (SNR)
Resource allocation (RBs)
Modulation scheme (MCS)
Tree-based models outperform linear models due to non-linear network behavior
Different 5G slices exhibit distinct QoS characteristics

🏆 Results
Achieved R² score of 0.94 using Random Forest
Successfully built a real-time QoS prediction system
Implemented interactive AI dashboard
Enabled proactive network optimization

🔮 Future Scope
Integration with real-time telecom data
Deployment in edge computing environments
Advanced explainability using SHAP
Integration with network orchestration systems

📌 Conclusion
This project demonstrates how AI can transform 5G network management from reactive to proactive by predicting QoS and enabling intelligent decision-making.

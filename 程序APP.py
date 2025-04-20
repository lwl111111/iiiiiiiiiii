import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap  # Import SHAP library
import matplotlib.pyplot as plt

# Load the pre-trained XGBoost model
model = joblib.load('best_model_catboost.pkl')

# Updated feature ranges with 'Previous high blood pressure' as categorical
feature_ranges = {
    "Age": {"type": "numerical"},
    "Previous high blood pressure": {"type": "categorical", "options": [0, 1]},
    "Systolic Blood Pressure": {"type": "numerical"},
    "Monocyte count": {"type": "numerical"},
    "Glycated haemoglobin HbA1c": {"type": "numerical"},
    "Plasma GDF15": {"type": "numerical"},
    "Plasma BCAN": {"type": "numerical"},
    "Plasma HAVCR1": {"type": "numerical"},
    "Plasma WFDC2": {"type": "numerical"},
    "Plasma NEFL": {"type": "numerical"},
    "Plasma MMP12": {"type": "numerical"}
}

# Streamlit interface title
st.title("10-Year Stroke Risk Prediction")

# Create a single column for input
col = st.container()

# Initialize an empty list to store feature values
feature_values = []

# Create input fields for each feature
for i, (feature, properties) in enumerate(feature_ranges.items()):
    if properties["type"] == "numerical":
        # For numerical inputs
        value = st.number_input(
            label=f"{feature}",
            value=0.0,  # Default value is 0
            key=f"{feature}_input"
        )
    elif properties["type"] == "categorical":
        # For categorical input (only "Previous high blood pressure")
        value = st.radio(
            label="Previous high blood pressure",
            options=[0, 1],  # 0 = No, 1 = Yes
            format_func=lambda x: "No" if x == 0 else "Yes",
            key=f"{feature}_input"
        )
    
    # Append the value to the list
    feature_values.append(value)

# 将特征值转换为模型输入格式
features = np.array([feature_values])

# 预测与SHAP可视化
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果，使用Matplotlib渲染指定字体
    text = f"Predicted probability of MACE in the next 10 years: {probability:.2f}%."
    fig, ax = plt.subplots(figsize=(10, 1))
    ax.text(
        0.5, 0.1, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',  # 使用Times New Roman字体
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.subplots_adjust(left=0.05, right=0.95, top=1, bottom=0)  # Adjust margins tightly
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=1200)
    st.image("prediction_text.png")

    # 计算SHAP值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    # 打印SHAP值的形状，确保它是二维的
    print(f"SHAP values shape: {np.shape(shap_values)}")
    
    # 获取Class 1（正类）的SHAP值，二分类问题时，shap_values会包含两个元素，分别对应两个类别
    shap_values_class_1 = shap_values  # 获取正类（Class 1）的SHAP值
    expected_value_class_1 = explainer.expected_value  # 获取Class 1的期望值

    # 计算每个特征的绝对SHAP值，按降序排序，选择前6个特征
    shap_values_abs = np.abs(shap_values_class_1).mean(axis=0)
    top_6_indices = np.argsort(shap_values_abs)[-6:]  # 获取前6个特征的索引

    # 只保留前6个特征的SHAP值和特征名称
    shap_values_class_1_top_6 = shap_values_class_1[:, top_6_indices]
    top_6_features = pd.DataFrame([feature_values], columns=feature_ranges.keys()).iloc[:, top_6_indices]

    # 生成仅显示前6个特征的SHAP力图
    shap_fig = shap.force_plot(
        expected_value_class_1,  # 使用Class 1的期望值
        shap_values_class_1_top_6,  # 前6个特征的SHAP值
        top_6_features,  # 仅显示前6个特征的数据
        matplotlib=True,
    )

    # 保存并显示SHAP力图
    plt.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0)  # Reduce bottom margin, adjust top
    plt.savefig("shap_force_plot_class_1_top_6.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot_class_1_top_6.png")




# ---------------------------
# 1. Обучение модели и сохранение
# ---------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from joblib import dump

# Загрузка данных
# Предполагается, что файл 'nanoparticles_dataset.csv' уже существует
# и содержит: size, zeta, coating, pH, enzyme_activity, efficacy

data = pd.read_csv("nanoparticles_dataset.csv")
X = pd.get_dummies(data.drop("efficacy", axis=1))
y = data["efficacy"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Сохраняем модель и scaler
dump(model, "nanoparticle_model.joblib")
dump(scaler, "scaler.joblib")

# ---------------------------
# 2. Streamlit-дэшборд (app.py)
# ---------------------------
# Запускается через: streamlit run app.py
import streamlit as st
import numpy as np
from joblib import load
from skopt import gp_minimize
from skopt.space import Real, Categorical

model = load("nanoparticle_model.joblib")
scaler = load("scaler.joblib")

st.title("\U0001F9E0 Дизайн умных наночастиц для таргетной доставки")

size = st.slider("Размер НЧ (нм)", 10, 200, 100)
zeta = st.slider("Zeta-потенциал (мВ)", -50, 50, 0)
coating = st.selectbox("Тип покрытия", ["PEG", "PLA", "lipid"])
pH = st.slider("Уровень pH", 5.5, 7.4, 6.8)
enzyme_activity = st.slider("Активность ферментов (0–1)", 0.0, 1.0, 0.5)

input_data = pd.DataFrame([{
    "size": size,
    "zeta": zeta,
    "coating": coating,
    "pH": pH,
    "enzyme_activity": enzyme_activity
}])

input_data = pd.get_dummies(input_data)
for col in model.feature_names_in_:
    if col not in input_data.columns:
        input_data[col] = 0
input_data = input_data[model.feature_names_in_]

scaled = scaler.transform(input_data)
prediction = model.predict(scaled)[0]

st.markdown(f"### \U0001F52E Прогнозируемая эффективность: **{prediction:.2f}**")

if st.button("\U0001F50D Найти оптимальный состав"):
    space = [
        Real(10, 200, name='size'),
        Real(-50, 50, name='zeta'),
        Categorical(['PEG', 'PLA', 'lipid'], name='coating'),
        Real(5.5, 7.4, name='pH'),
        Real(0.0, 1.0, name='enzyme_activity')
    ]

    def objective(params):
        df = pd.DataFrame([params], columns=['size', 'zeta', 'coating', 'pH', 'enzyme_activity'])
        df = pd.get_dummies(df)
        for col in model.feature_names_in_:
            if col not in df.columns:
                df[col] = 0
        df = df[model.feature_names_in_]
        scaled = scaler.transform(df)
        return -model.predict(scaled)[0]

    result = gp_minimize(objective, space, n_calls=25)

    st.markdown("### \U0001F9EA Оптимальные параметры НЧ:")
    names = ["size", "zeta", "coating", "pH", "enzyme_activity"]
    for name, val in zip(names, result.x):
        st.write(f"**{name}**: {val}")
    st.markdown(f"### \U0001F680 Ожидаемая эффективность: **{ -result.fun:.2f}**")

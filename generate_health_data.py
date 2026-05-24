#!/usr/bin/env python3
"""
Script para gerar dados de saúde pessoal (Personal Health Dataset)
Cria arquivo CSV com features de saúde para ablation study
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_personal_health_data(num_records=5000, output_path='./data/personal_health_data.csv'):
    """
    Gera dados de saúde pessoal para o dataset Health
    
    Colunas esperadas:
    - User_ID: ID do usuário
    - Timestamp: Data/hora
    - Heart_Rate: Frequência cardíaca (60-120 bpm)
    - Blood_Pressure_Systolic: Pressão sistólica (90-180)
    - Blood_Pressure_Diastolic: Pressão diastólica (60-120)
    - Blood_Glucose: Glicose no sangue (70-180)
    - SpO2: Saturação de oxigênio (95-100%)
    - Temperature: Temperatura corporal (36-39°C)
    - Sleep_Hours: Horas de sono (0-12)
    - Physical_Activity: Atividade física (0-100 minutos)
    - Stress_Level: Nível de estresse (0-100)
    - Water_Intake: Ingestão de água (0-4 litros)
    - Diet_Quality: Qualidade da dieta (0-100)
    - Exercise_Frequency: Frequência de exercício (0-7 dias/semana)
    - Medical_Conditions: Condições médicas (text)
    - Alcohol_Consumption: Consumo de álcool (None/Low/Medium/High)
    - Medication_Use: Uso de medicação (Yes/No)
    - Smoking_Status: Status de fumo (Never/Former/Current)
    - Depression_Risk: Risco de depressão (Low/Medium/High)
    - Anxiety_Risk: Risco de ansiedade (Low/Medium/High)
    - Sleep_Quality: Qualidade do sono (Poor/Fair/Good/Excellent)
    - Disease_Risk: Risco de doença (Low/Medium/High)
    - Hypertension_Risk: Risco de hipertensão (Low/Medium/High)
    - Diabetes_Risk: Risco de diabetes (Low/Medium/High)
    - Cardiovascular_Risk: Risco cardiovascular (Low/Medium/High)
    - Mental_Health_Score: Pontuação de saúde mental (0-100)
    - Physical_Health_Score: Pontuação de saúde física (0-100)
    - Overall_Wellness: Bem-estar geral (Poor/Fair/Good/Excellent)
    - Anomaly_Flag: Se há anomalia (0/1) - Label
    """
    
    np.random.seed(42)
    
    # Gerar dados
    user_ids = np.arange(1, num_records + 1)
    
    base_date = datetime(2024, 1, 1)
    timestamps = [base_date + timedelta(hours=i) for i in range(num_records)]
    
    # Features numéricas
    heart_rate = np.random.normal(75, 15, num_records)
    heart_rate = np.clip(heart_rate, 40, 150)
    
    bp_sys = np.random.normal(120, 20, num_records)
    bp_sys = np.clip(bp_sys, 80, 200)
    
    bp_dias = np.random.normal(80, 15, num_records)
    bp_dias = np.clip(bp_dias, 50, 120)
    
    glucose = np.random.normal(100, 20, num_records)
    glucose = np.clip(glucose, 70, 200)
    
    spo2 = np.random.normal(97, 2, num_records)
    spo2 = np.clip(spo2, 90, 100)
    
    temperature = np.random.normal(37, 0.5, num_records)
    temperature = np.clip(temperature, 35, 40)
    
    sleep_hours = np.random.gamma(2, 3, num_records)
    sleep_hours = np.clip(sleep_hours, 0, 12)
    
    physical_activity = np.random.exponential(30, num_records)
    physical_activity = np.clip(physical_activity, 0, 300)
    
    stress_level = np.random.normal(40, 25, num_records)
    stress_level = np.clip(stress_level, 0, 100)
    
    water_intake = np.random.gamma(2, 0.5, num_records)
    water_intake = np.clip(water_intake, 0, 4)
    
    diet_quality = np.random.normal(60, 20, num_records)
    diet_quality = np.clip(diet_quality, 0, 100)
    
    exercise_freq = np.random.poisson(3, num_records)
    exercise_freq = np.clip(exercise_freq, 0, 7)
    
    mental_health_score = np.random.normal(70, 15, num_records)
    mental_health_score = np.clip(mental_health_score, 0, 100)
    
    physical_health_score = np.random.normal(75, 15, num_records)
    physical_health_score = np.clip(physical_health_score, 0, 100)
    
    # Features categóricas
    medical_conditions = np.random.choice(
        ['None', 'Hypertension', 'Diabetes', 'Asthma', 'Thyroid', 'Arthritis', 'Heart_Disease'],
        num_records, p=[0.5, 0.15, 0.15, 0.1, 0.05, 0.03, 0.02]
    )
    
    alcohol_consumption = np.random.choice(
        ['None', 'Low', 'Medium', 'High'],
        num_records, p=[0.5, 0.3, 0.15, 0.05]
    )
    
    medication_use = np.random.choice(['Yes', 'No'], num_records, p=[0.3, 0.7])
    
    smoking_status = np.random.choice(
        ['Never', 'Former', 'Current'],
        num_records, p=[0.6, 0.25, 0.15]
    )
    
    depression_risk = np.random.choice(['Low', 'Medium', 'High'], num_records, p=[0.7, 0.2, 0.1])
    anxiety_risk = np.random.choice(['Low', 'Medium', 'High'], num_records, p=[0.65, 0.25, 0.1])
    
    sleep_quality = np.random.choice(
        ['Poor', 'Fair', 'Good', 'Excellent'],
        num_records, p=[0.15, 0.25, 0.35, 0.25]
    )
    
    disease_risk = np.random.choice(['Low', 'Medium', 'High'], num_records, p=[0.6, 0.3, 0.1])
    hypertension_risk = np.random.choice(['Low', 'Medium', 'High'], num_records, p=[0.65, 0.25, 0.1])
    diabetes_risk = np.random.choice(['Low', 'Medium', 'High'], num_records, p=[0.7, 0.2, 0.1])
    cardiovascular_risk = np.random.choice(['Low', 'Medium', 'High'], num_records, p=[0.6, 0.3, 0.1])
    
    overall_wellness = np.random.choice(
        ['Poor', 'Fair', 'Good', 'Excellent'],
        num_records, p=[0.1, 0.2, 0.4, 0.3]
    )
    
    # Gerar Anomaly_Flag (target label)
    # Anomalias ocorrem quando múltiplos sinais estão fora do intervalo normal
    anomaly_flags = []
    for i in range(num_records):
        anomaly_score = 0
        
        # Pontuação baseada em features
        if heart_rate[i] > 100 or heart_rate[i] < 60:
            anomaly_score += 1
        if bp_sys[i] > 160 or bp_sys[i] < 90:
            anomaly_score += 1
        if glucose[i] > 150 or glucose[i] < 80:
            anomaly_score += 1
        if spo2[i] < 95:
            anomaly_score += 1
        if sleep_hours[i] < 5:
            anomaly_score += 1
        if stress_level[i] > 70:
            anomaly_score += 1
        if mental_health_score[i] < 50:
            anomaly_score += 1
        if physical_health_score[i] < 50:
            anomaly_score += 1
        if disease_risk[i] == 'High':
            anomaly_score += 2
        
        # Anomalia se score >= 4
        anomaly_flags.append(1 if anomaly_score >= 4 else 0)
    
    anomaly_flags = np.array(anomaly_flags)
    
    # Criar DataFrame
    df = pd.DataFrame({
        'User_ID': user_ids,
        'Timestamp': timestamps,
        'Heart_Rate': heart_rate,
        'Blood_Pressure_Systolic': bp_sys,
        'Blood_Pressure_Diastolic': bp_dias,
        'Blood_Glucose': glucose,
        'SpO2': spo2,
        'Temperature': temperature,
        'Sleep_Hours': sleep_hours,
        'Physical_Activity': physical_activity,
        'Stress_Level': stress_level,
        'Water_Intake': water_intake,
        'Diet_Quality': diet_quality,
        'Exercise_Frequency': exercise_freq,
        'Medical_Conditions': medical_conditions,
        'Alcohol_Consumption': alcohol_consumption,
        'Medication_Use': medication_use,
        'Smoking_Status': smoking_status,
        'Depression_Risk': depression_risk,
        'Anxiety_Risk': anxiety_risk,
        'Sleep_Quality': sleep_quality,
        'Disease_Risk': disease_risk,
        'Hypertension_Risk': hypertension_risk,
        'Diabetes_Risk': diabetes_risk,
        'Cardiovascular_Risk': cardiovascular_risk,
        'Mental_Health_Score': mental_health_score,
        'Physical_Health_Score': physical_health_score,
        'Overall_Wellness': overall_wellness,
        'Anomaly_Flag': anomaly_flags
    })
    
    # Criar pasta data se não existir
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Salvar CSV
    df.to_csv(output_path, index=False)
    
    print(f"✓ Health dataset gerado: {output_path}")
    print(f"  - Total registros: {num_records}")
    print(f"  - Total features: {len(df.columns) - 1} (excluindo Anomaly_Flag)")
    print(f"  - Anomalias: {anomaly_flags.sum()} ({100*anomaly_flags.sum()/num_records:.1f}%)")
    print(f"\nDataFrame Info:")
    print(f"  Shape: {df.shape}")
    print(f"\nPrimeiras 5 linhas:")
    print(df.head())
    print(f"\nÚltimas 5 linhas:")
    print(df.tail())
    
    return df

if __name__ == "__main__":
    # Gerar dataset
    df = generate_personal_health_data(num_records=5000)
    print("\n✓ Pronto para rodar ablation study com dataset Health!")

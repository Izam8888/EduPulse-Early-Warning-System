# EduPulse

EduPulse adalah **Student Early Warning System** berbasis Machine Learning untuk membantu mengidentifikasi risiko dropout siswa sejak dini.

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/Izam8888/edupulse-streamlit-dropout-risk.git
cd edupulse-streamlit-dropout-risk
```

### 2. Buat Virtual Environment

Windows:

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi

```bash
streamlit run app.py
```

## Features

### Dashboard
- Total students
- Actual dropout rate
- Students at risk
- Risk distribution
- Correlation analysis
- Relationship analysis

### Student Risk Monitoring
- Filter High / Medium / Low Risk
- Search Student ID
- Pagination
- Download monitoring data

### Student Prediction
Menggunakan 6 indikator:
- GPA
- Stress Index
- Attendance Rate
- Study Hours / Day
- Travel Time
- Assignment Delay

Output:
- Dropout Probability
- Risk Level
- Risk Threshold
- Recommendations

## Machine Learning

Model utama:

**Lightweight Logistic Regression**

Features:
- `GPA`
- `Stress_Index`
- `Attendance_Rate`
- `Study_Hours_per_Day`
- `Travel_Time_Minutes`
- `Assignment_Delay_Days`

### Test Set Performance

| Metric | Score |
|---|---:|
| Accuracy | 0.7385 |
| Precision | 0.4660 |
| Recall | 0.7558 |
| F1-score | 0.5765 |
| ROC-AUC | 0.8166 |

## Risk Threshold

| Risk Level | Probability |
|---|---:|
| Low Risk | < 40% |
| Medium Risk | 40% – < 70% |
| High Risk | >= 70% |

## Dataset

**Student Dropout Prediction Dataset**

Source:  
https://www.kaggle.com/datasets/meharshanali/student-dropout-prediction-dataset

Dataset yang digunakan terdiri dari **10.000 data siswa dan 19 kolom**.

Target:
- `0` = Not Dropout
- `1` = Dropout

## Project Structure

```text
edupulse-streamlit-2/
├── app.py
├── predict.py
├── edupulse_dropout_model.joblib
├── student_dropout_dataset_v3.csv
├── requirements.txt
├── README.md
└── new-logo-edupulse/
    └── new-logo-edupulse.svg
```

## Model Integration

`predict.py` menyediakan fungsi:

```python
predict_dropout(student_data)
```

Artifact model:

```text
edupulse_dropout_model.joblib
```

Artifact menyimpan model, preprocessing, feature list, target, dan risk thresholds.

## Team

**Dicoding Capstone Project — DB14-G001**

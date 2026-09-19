# EduPulse

EduPulse adalah **Student Early Warning System** berbasis Machine Learning untuk membantu memprediksi nilai ujian akhir siswa dan mengidentifikasi siswa yang membutuhkan perhatian akademik lebih awal.

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/Izam8888/EduPulse-Early-Warning-System.git

cd EduPulse-Early-Warning-System
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
- Average Final Exam
- Students at risk
- Average attendance
- Risk distribution

### Student Prediction

Menggunakan 5 indikator:

- Attendance
- Quiz 1 Score
- Quiz 2 Score
- Assignment Score
- Daily Study Hours

Output:

- Predicted Final Exam Score
- Risk Level
- Risk Threshold
- Academic Recommendations

## Machine Learning

Model utama:

**Linear Regression**

Features:

- `attendance`
- `quiz_1_score_pct`
- `quiz_2_score_pct`
- `assignment_pct`
- `daily_study_hours`

### Test Set Performance

| Metric | Score |
|---|---:|
| MAE | 3.675 |
| RMSE | 4.618 |
| R² | 0.830 |

## Risk Threshold

| Risk Level | Predicted Final Exam |
|---|---:|
| High Risk | < 70 |
| Medium Risk | 70 – < 80 |
| Low Risk | >= 80 |

## Dataset

**Student Academic Performance Dataset**

Source:

https://www.kaggle.com/datasets/sonalshinde123/student-academic-performance-dataset

License:

**CC0: Public Domain**

Dataset yang digunakan terdiri dari **2.000 data siswa dan 7 kolom**.

Variabel:

- `Student_ID`
- `Attendance (%)`
- `Internal Test 1 (out of 40)`
- `Internal Test 2 (out of 40)`
- `Assignment Score (out of 10)`
- `Daily Study Hours`
- `Final Exam Marks (out of 100)`

Target:

- `Final Exam Marks (out of 100)`

## Project Structure

```text
EduPulse-Early-Warning-System/
├── app.py
├── predict.py
├── edupulse_model_final.joblib
├── Final_Marks_Data.csv
├── requirements.txt
├── README.md
├── notebooks/
│   ├── EduPulse_Final_Score.ipynb
│   ├── edupulse_model_final.joblib
│   └── predict.py
├── new-logo-edupulse/
│   └── new-logo-edupulse.svg
└── dropout-risk/
    ├── app.py
    ├── predict.py
    ├── edupulse_dropout_model.joblib
    ├── student_dropout_dataset_v3.csv
    ├── requirements.txt
    ├── README.md
    └── notebooks/
        ├── EduPulse_Dropout_Risk.ipynb
        ├── edupulse_dropout_model (8).joblib
        └── predict (9).py
```

## Model Integration

`predict.py` digunakan untuk melakukan prediksi nilai Final Exam berdasarkan fitur input yang tersedia.

Artifact model:

```text
edupulse_model_final.joblib
```

Artifact menyimpan model **Linear Regression**, feature list, risk thresholds, model version, dan evaluation metrics.

## Team

**Dicoding Capstone Project — DB14-G001**
# Sleep Disorder Predictor — COM572 Coursework Task 1

**Module:** COM572 – Machine Learning  
**Task:** End-to-end ML application with Streamlit deployment  
**Target:** Predict Sleep Disorder (None / Insomnia / Sleep Apnoea)

---

## Project Structure

```
├── app.py                                   # Streamlit web application
├── train.py                                 # Standalone training & evaluation script
├── Sleep_health_and_lifestyle_dataset.csv   # Dataset (must be in the same directory)
├── requirements.txt                         # Python dependencies
└── README.md                                # This file
```

---

## Getting Started

### 1. Clone / Download the repository

Ensure all files are in the **same directory**.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the standalone training script (optional)

This trains and evaluates all three models, reports metrics, and performs hyperparameter tuning.

```bash
python train.py
```

### 4. Launch the Streamlit application

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## Dataset

**Name:** Sleep Health and Lifestyle Dataset  
**Source:** [Kaggle – lakshminarayanan lakshmi](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)  
**Records:** 374  
**Target classes:** None (219) · Insomnia (77) · Sleep Apnea (78)

---

## Models Implemented

| Model | Purpose |
|---|---|
| Logistic Regression | Linear baseline |
| Decision Tree | Interpretable non-linear model |
| Random Forest (tuned) | Final deployed model |

---

## Key Results

| Model | Test Accuracy | Weighted F1 | CV F1 |
|---|---|---|---|
| Logistic Regression | 93.3% | 0.934 | 0.883 |
| Decision Tree | 88.0% | 0.884 | 0.842 |
| Random Forest (default) | 92.0% | 0.921 | 0.855 |
| **Random Forest (tuned)** | **96.0%** | **0.960** | **0.874** |

---

## Disclaimer

This application is intended for **educational purposes only**. It does not constitute medical advice. Always consult a qualified healthcare professional regarding sleep health concerns.

# Airbnb Nightly Price Prediction - Team Overfitters

**YZV 311E - Data Mining (Fall 2025-2026)** **Istanbul Technical University**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-yellow)
![Status](https://img.shields.io/badge/Status-Active-green)

## Project Overview
This project is developed for the **YZV 311E Data Mining** course competition. [cite_start]The main objective is to predict the nightly price of Airbnb listings based on a diverse set of features including host details, property descriptions, availability, and customer reviews[cite: 3, 4].

The project focuses on the complete **Data Mining Pipeline**:
1.  **Data Understanding & EDA:** Analyzing distributions, correlations, and outliers.
2.  **Preprocessing:** Handling missing values, cleaning text data, and formatting prices.
3.  **Feature Engineering:** Creating interpretable features from text (NLP), dates, and geospatial data.
4.  **Modeling:** Implementing baseline regressions and advanced ensemble methods (XGBoost/LightGBM).
5.  **Evaluation:** Optimizing for **RMSLE** (Root Mean Squared Logarithmic Error).

## 👥 Team Members (Team Overfitters)
* **İbrahim Bancar** - Data Exploration & Preprocessing
* **Hasan Kan** - Feature Engineering & Modeling
* **Alperen Sağlam** - Evaluation & Reporting

## Repository Structure
```text
├── data/                  # Raw and processed data (Not included in git)
│   ├── train.csv
│   ├── test.csv
│   └── ...
├── notebooks/             # Jupyter Notebooks for experiments
│   ├── 01_EDA_Preprocessing.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   └── 03_Baseline_Model.ipynb
├── src/                   # Source code scripts (if applicable)
├── submissions/           # Kaggle submission files
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
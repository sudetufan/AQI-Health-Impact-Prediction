# AQI Health Impact Prediction

## Project Overview

This project uses Machine Learning techniques to analyze air pollution data and predict:

1. Air Quality Category
   - Good
   - Moderate
   - Poor
   - Hazardous

2. Health Risk Level
   - Very Low
   - Low
   - Moderate
   - High
   - Very High

The project is implemented using Python and Scikit-Learn with Random Forest classifiers.

---

## Datasets

### Dataset 1
Air pollution measurements used for Air Quality Classification.

Features:
- Temperature
- Humidity
- PM2.5
- PM10
- NO2
- SO2
- CO
- Proximity to Industrial Areas
- Population Density

Target:
- Air Quality

### Dataset 2
Air pollution and environmental measurements used for Health Risk Prediction.

Features:
- PM10
- PM2.5
- NO2
- SO2
- O3
- Temperature
- Humidity
- Wind Speed

Target:
- Health Impact Class (0–4)

---

## Machine Learning Models

### Model A
Air Quality Classification

Algorithm:
- Random Forest Classifier

### Model B
Health Risk Classification

Techniques:
- Random Forest Classifier
- SMOTE
- Class Weight Balancing

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Imbalanced-Learn
- Joblib

---



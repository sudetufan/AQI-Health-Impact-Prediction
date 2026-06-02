"""
Model A — Air Quality Classification (DS1)
Dataset: updated_pollution_dataset.csv
Target : Air Quality → Good / Moderate / Poor / Hazardous
Features: Temperature, Humidity, PM2.5, PM10, NO2, SO2, CO,
          Proximity_to_Industrial_Areas, Population_Density
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), '..', 'data', 'updated_pollution_dataset.csv')
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def train_model_a():
    # 1. Veri yükleme
    df = pd.read_csv(DATA_PATH)
    print(f"[Model A] Veri yüklendi: {df.shape}")
    print(f"  Sınıf dağılımı:\n{df['Air Quality'].value_counts()}\n")

    # 2. Feature / Target ayrımı
    X = df.drop(columns=['Air Quality'])
    le = LabelEncoder()
    y = le.fit_transform(df['Air Quality'])
    # Sınıf sırası: Good=0, Hazardous=1, Moderate=2, Poor=3

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Scaling
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # 5. Model eğitimi
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_sc, y_train)

    # 6. Cross-validation
    cv_scores = cross_val_score(model, X_train_sc, y_train, cv=5, scoring='accuracy')
    print(f"[Model A] CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 7. Test değerlendirmesi
    y_pred = model.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)
    print(f"[Model A] Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # 8. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Model A — Confusion Matrix')
    plt.ylabel('Gerçek')
    plt.xlabel('Tahmin')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_a_confusion_matrix.png'), dpi=150)
    plt.close()

    # 9. Feature Importance
    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
    plt.figure(figsize=(7, 5))
    fi.plot(kind='barh', color='steelblue')
    plt.title('Model A — Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_a_feature_importance.png'), dpi=150)
    plt.close()

    # 10. Modeli kaydet
    joblib.dump(model,  os.path.join(OUTPUT_DIR, 'model_a.pkl'))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'model_a_scaler.pkl'))
    joblib.dump(le,     os.path.join(OUTPUT_DIR, 'model_a_label_encoder.pkl'))
    print("[Model A] model_a.pkl kaydedildi.\n")

    return model, scaler, le


if __name__ == '__main__':
    train_model_a()

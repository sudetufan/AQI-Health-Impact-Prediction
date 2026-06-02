"""
Model B — Health Impact Classification (DS2)
Dataset: air_quality_health_impact_data.csv
Target : HealthImpactClass → 0 (Çok Düşük) … 4 (Çok Yüksek risk)
Features: PM10, PM2_5, NO2, SO2, O3, Temperature, Humidity, WindSpeed

NOTLAR:
- AQI sütunu KULLANILMAZ (diğer feature'larla korelasyonu ~0, sentetik üretilmiş)
- Sınıf dengesizliği var → class_weight='balanced' + SMOTE
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.join(os.path.dirname(__file__), '..', 'data', 'air_quality_health_impact_data.csv')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_COLS = ['PM10', 'PM2_5', 'NO2', 'SO2', 'O3',
                'Temperature', 'Humidity', 'WindSpeed']

RISK_LABELS = {
    0: 'Çok Düşük',
    1: 'Düşük',
    2: 'Orta',
    3: 'Yüksek',
    4: 'Çok Yüksek'
}

def train_model_b():
    # 1. Veri yükleme
    df = pd.read_csv(DATA_PATH)
    print(f"[Model B] Veri yüklendi: {df.shape}")
    print(f"  Sınıf dağılımı (önce SMOTE):\n{df['HealthImpactClass'].value_counts()}\n")

    X = df[FEATURE_COLS]
    y = df['HealthImpactClass'].astype(int)

    # 2. Train/test split (SMOTE sadece train'e uygulanır)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Scaling
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    # 4. SMOTE — sınıf dengesizliğini gider
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_sc, y_train)
    print(f"  Sınıf dağılımı (sonra SMOTE): {dict(zip(*np.unique(y_train_res, return_counts=True)))}\n")

    # 5. Model eğitimi
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_res, y_train_res)

    # 6. Cross-validation (balanced verisiyle)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_res, y_train_res, cv=cv, scoring='accuracy')
    print(f"[Model B] CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # 7. Test değerlendirmesi
    y_pred = model.predict(X_test_sc)
    acc = accuracy_score(y_test, y_pred)
    print(f"[Model B] Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred,
          target_names=[RISK_LABELS[i] for i in sorted(RISK_LABELS)]))

    # 8. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=list(RISK_LABELS.values()),
                yticklabels=list(RISK_LABELS.values()))
    plt.title('Model B — Confusion Matrix')
    plt.ylabel('Gerçek')
    plt.xlabel('Tahmin')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_b_confusion_matrix.png'), dpi=150)
    plt.close()

    # 9. Feature Importance
    fi = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=True)
    plt.figure(figsize=(7, 5))
    fi.plot(kind='barh', color='darkorange')
    plt.title('Model B — Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'model_b_feature_importance.png'), dpi=150)
    plt.close()

    # 10. Modeli kaydet
    joblib.dump(model,  os.path.join(OUTPUT_DIR, 'model_b.pkl'))
    joblib.dump(scaler, os.path.join(OUTPUT_DIR, 'model_b_scaler.pkl'))
    print("[Model B] model_b.pkl kaydedildi.\n")

    return model, scaler


if __name__ == '__main__':
    train_model_b()

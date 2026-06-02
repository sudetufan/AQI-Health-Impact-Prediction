"""
predict.py — Eğitilmiş modelleri kullanarak yeni veri tahmini

Kullanım:
    python models/predict.py
"""

import numpy as np
import joblib
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs')

RISK_LABELS = {
    0: 'Cok Dusuk',
    1: 'Dusuk',
    2: 'Orta',
    3: 'Yuksek',
    4: 'Cok Yuksek'
}


def predict_air_quality(temperature, humidity, pm25, pm10,
                        no2, so2, co, proximity_industrial, population_density):
    """
    Model A: Hava kalitesi sınıfı tahmini

    Parametreler:
        temperature          : Sıcaklık (°C)
        humidity             : Nem (%)
        pm25                 : PM2.5 (µg/m³)
        pm10                 : PM10  (µg/m³)
        no2                  : NO2   (µg/m³)
        so2                  : SO2   (µg/m³)
        co                   : CO    (mg/m³)
        proximity_industrial : Endüstriyel alana yakınlık (km)
        population_density   : Nüfus yoğunluğu (kişi/km²)

    Döndürür:
        str: 'Good' | 'Moderate' | 'Poor' | 'Hazardous'
    """
    model  = joblib.load(os.path.join(OUTPUT_DIR, 'model_a.pkl'))
    scaler = joblib.load(os.path.join(OUTPUT_DIR, 'model_a_scaler.pkl'))
    le     = joblib.load(os.path.join(OUTPUT_DIR, 'model_a_label_encoder.pkl'))

    import pandas as pd
    features = pd.DataFrame([[temperature, humidity, pm25, pm10,
                               no2, so2, co, proximity_industrial, population_density]],
                             columns=['Temperature','Humidity','PM2.5','PM10',
                                      'NO2','SO2','CO','Proximity_to_Industrial_Areas',
                                      'Population_Density'])
    scaled = scaler.transform(features)
    pred_idx = model.predict(scaled)[0]
    proba    = model.predict_proba(scaled)[0]
    label    = le.inverse_transform([pred_idx])[0]

    print(f"\n[Model A] Hava Kalitesi Tahmini: {label}")
    print("  Olasılıklar:")
    for cls, p in zip(le.classes_, proba):
        bar = '#' * int(p * 30)
        print(f"    {cls:<12} {bar} {p*100:.1f}%")

    return label


def predict_health_risk(pm10, pm2_5, no2, so2, o3,
                        temperature, humidity, wind_speed):
    """
    Model B: Sağlık etki seviyesi tahmini

    Parametreler:
        pm10        : PM10       (µg/m³)
        pm2_5       : PM2.5      (µg/m³)
        no2         : NO2        (µg/m³)
        so2         : SO2        (µg/m³)
        o3          : Ozon O3    (µg/m³)
        temperature : Sıcaklık   (°C)
        humidity    : Nem        (%)
        wind_speed  : Rüzgar hızı (m/s)

    Döndürür:
        int: 0–4 arası risk sınıfı
    """
    model  = joblib.load(os.path.join(OUTPUT_DIR, 'model_b.pkl'))
    scaler = joblib.load(os.path.join(OUTPUT_DIR, 'model_b_scaler.pkl'))

    import pandas as pd
    COLS = ['PM10','PM2_5','NO2','SO2','O3','Temperature','Humidity','WindSpeed']
    features = pd.DataFrame([[pm10, pm2_5, no2, so2, o3,
                               temperature, humidity, wind_speed]], columns=COLS)
    scaled = scaler.transform(features)
    pred     = model.predict(scaled)[0]
    proba    = model.predict_proba(scaled)[0]

    print(f"\n[Model B] Sağlık Risk Seviyesi: {RISK_LABELS[pred]} (Sınıf {pred})")
    print("  Olasılıklar:")
    for i, p in enumerate(proba):
        bar = '#' * int(p * 30)
        print(f"    Sinif {i} {RISK_LABELS[i]:<18} {bar} {p*100:.1f}%")

    return pred


# ── Örnek kullanım ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("ÖRNEK 1 — Düşük kirlilik (temiz hava beklentisi)")
    print("=" * 55)
    predict_air_quality(
        temperature=22, humidity=55, pm25=4, pm10=15,
        no2=15, so2=5, co=1.2,
        proximity_industrial=10, population_density=200
    )
    predict_health_risk(
        pm10=15, pm2_5=4, no2=15, so2=5, o3=30,
        temperature=22, humidity=55, wind_speed=8
    )

    print("\n" + "=" * 55)
    print("ÖRNEK 2 — Yüksek kirlilik (kentsel/endüstriyel)")
    print("=" * 55)
    predict_air_quality(
        temperature=35, humidity=80, pm25=60, pm10=150,
        no2=55, so2=40, co=3.5,
        proximity_industrial=1.5, population_density=900
    )
    predict_health_risk(
        pm10=150, pm2_5=60, no2=55, so2=40, o3=90,
        temperature=35, humidity=80, wind_speed=2
    )

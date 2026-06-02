"""
main.py — AQI & Health Impact ML Pipeline
Çalıştır: python main.py
"""

import sys
import os

# models/ klasörünü import path'e ekle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'models'))

from train_model_a import train_model_a
from train_model_b import train_model_b
from predict import predict_air_quality, predict_health_risk


def main():
    print("=" * 60)
    print("  AQI & HEALTH IMPACT ML PIPELINE")
    print("=" * 60)

    # 1. Model A eğit
    print("\n[*] Model A egitiliyor (Air Quality Classification)...")
    train_model_a()

    # 2. Model B eğit
    print("[*] Model B egitiliyor (Health Impact Classification)...")
    train_model_b()

    # 3. Örnek tahminler
    print("\n" + "=" * 60)
    print("  TAHMIN ORNEKLERI")
    print("=" * 60)

    print("\n[1] Senaryo 1 -- Temiz Hava")
    predict_air_quality(
        temperature=22, humidity=55, pm25=4, pm10=15,
        no2=15, so2=5, co=1.2,
        proximity_industrial=10, population_density=200
    )
    predict_health_risk(
        pm10=15, pm2_5=4, no2=15, so2=5, o3=30,
        temperature=22, humidity=55, wind_speed=8
    )

    print("\n[2] Senaryo 2 -- Yuksek Kirlilik")
    predict_air_quality(
        temperature=35, humidity=80, pm25=60, pm10=150,
        no2=55, so2=40, co=3.5,
        proximity_industrial=1.5, population_density=900
    )
    predict_health_risk(
        pm10=150, pm2_5=60, no2=55, so2=40, o3=90,
        temperature=35, humidity=80, wind_speed=2
    )

    print("\n[OK] Pipeline tamamlandi.")
    print("[*] Kaydedilen dosyalar -> outputs/ klasoru:")
    for f in os.listdir('outputs'):
        print(f"   • {f}")


if __name__ == '__main__':
    main()

# AQI & Health Impact ML Project

## Proje Amacı
Hava kalitesi parametrelerinden:
- **Model A**: Hava kalitesi sınıfı tahmini (Good / Moderate / Poor / Hazardous)
- **Model B**: Sağlık etki seviyesi tahmini (0=Çok Düşük → 4=Çok Yüksek risk)

## Proje Yapısı
```
aqi_project/
├── CLAUDE.md              ← Bu dosya (Claude Code talimatları)
├── requirements.txt       ← Bağımlılıklar
├── data/
│   ├── updated_pollution_dataset.csv        ← DS1 (Model A)
│   └── air_quality_health_impact_data.csv   ← DS2 (Model B)
├── models/
│   ├── train_model_a.py   ← DS1 eğitim scripti
│   ├── train_model_b.py   ← DS2 eğitim scripti
│   └── predict.py         ← Yeni veri tahmini
├── outputs/               ← Kaydedilen modeller ve görseller
└── main.py                ← Her şeyi çalıştıran ana script
```

## Önemli Notlar (Claude Code için)
- DS1 ve DS2 **merge EDİLMEZ** — değer aralıkları ve ölçekleri tamamen farklı
- DS2'deki `AQI` sütunu feature olarak **kullanılmaz** — diğer sütunlarla korelasyonu ~0
- Model B'de sınıf dengesizliği var (sınıf 0 baskın) → `class_weight='balanced'` kullanılır
- Modeller eğitim sonrası `outputs/` klasörüne `.pkl` olarak kaydedilir

## Çalıştırma Sırası
```bash
pip install -r requirements.txt
python main.py
```

## Bağımlılıklar
scikit-learn, pandas, numpy, matplotlib, seaborn, joblib, imbalanced-learn

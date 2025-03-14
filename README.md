# CGE-product-counting-tool

Bu repository, ürün sayım işlemlerini geliştirmek ve test süreçlerini otomatikleştirmek amacıyla oluşturulmuştur. Projede, günlük kesim verilerinin temizlenmesinden dataset oluşturulmasına, günlük ürün kontrollerinden yeni işlev testlerine kadar çeşitli modüller yer almaktadır.

## İçindekiler
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Özellikler](#özellikler)
- [Kod Yapısı](#kod-yapısı)
- [Katkıda Bulunma](#katkıda-bulunma)
- [Lisans](#lisans)
- [İletişim](#iletişim)

```
proje_adi/
├── clear-dataset             # Test için hazırlanmış kesim verileri 
├── create_dataset            # Sistem tarafından günlük olarak oluşturulan JSON verisini kullanarak clear-dataset oluşturmak için kullanılan kodlar
├── daily-check-intermac-product  # Günlük ürün sayım ve kontrol işlemleri
└── new-function-test         # Yeni eklenen fonksiyonlar ile clear-dataset kullanarak performans testlerinin yapıldığı modül
```


## Dosya Açıklamaları

### clear-dataset
- **clean_data.py**: Kesim verilerini temizleyerek, analiz için uygun hale getirir. Verideki gereksiz veya hatalı kayıtları filtreler.
- **preprocess.py**: Temizlenmiş verilerin önişlemesini yapar; örneğin veri tiplerini dönüştürme, eksik değerleri işleme gibi adımları gerçekleştirir.

### create_dataset
- **generate_dataset.py**: Günlük olarak oluşturulan JSON verisini okuyarak, gerekli formatta dataset oluşturur.
- **merge_data.py**: Farklı kaynaklardan gelen verileri birleştirip, bütünleşik bir veri seti haline getirir.

### daily-check-intermac-product
- **product_check.py**: Günlük ürün sayım ve kontrol işlemlerini yürütür. Verilerin doğruluğunu ve güncelliğini kontrol eder.
- **report_generator.py**: Günlük kontroller sonrası rapor üretimi yapar; istatistiksel özetler ve görselleştirmeler sunar.

### new-function-test
- **performance_test.py**: Yeni eklenen fonksiyonların performans testlerini yapar. Test sonuçlarını kaydeder ve raporlar.
- **function_validator.py**: Yeni işlevlerin doğruluğunu ve verimliliğini kontrol eden modül.

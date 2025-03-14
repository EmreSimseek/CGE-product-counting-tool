import os
import json
import numpy as np
from scipy.spatial import procrustes
from scipy.interpolate import interp1d
from datetime import datetime
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
import logging
import threading

class ShapeMatcher:
    """
    Bu sınıf, Dinamik Zaman Sargılama (DTW) ve Pearson Korelasyon Katsayısını kullanarak,
    iki nokta serisinin benzerliğini değerlendirir ve birleşik bir skor hesaplar.
    """
    def __init__(self, tolerance_dtw=3000, tolerance_procrustes=0.06, tolerance_pearson=100, num_points=100):
        self.current_dir = os.getcwd()
        self.setup_logger_series()
        self.tolerance_dtw = tolerance_dtw
        self.tolerance_procrustes = tolerance_procrustes
        self.tolerance_pearson = tolerance_pearson
        self.num_points = num_points  # Normalize edilecek nokta sayısı
        # JSON dosyalarına eşzamanlı erişimi kontrol etmek için Lock
        self.json_lock = threading.Lock()

    def normalize_shape(self, series):
        """Nokta sayısını normalize et."""
        num_points_original = len(series)
        if num_points_original == self.num_points:
            return series  # Eğer zaten doğru sayıda nokta varsa, işlem yapma

        # Orijinal ve hedef indeksler oluştur
        original_indices = np.linspace(0, 1, num=num_points_original)
        target_indices = np.linspace(0, 1, num=self.num_points)

        # Her iki eksen için ara değerleme yap
        interpolator = interp1d(original_indices, series, axis=0, kind='cubic', fill_value="extrapolate")
        normalized_series = interpolator(target_indices)

        return normalized_series

    def check_for_nan_or_inf(self, series):
        """Dizide NaN veya Inf değerleri kontrol et."""
        if np.isnan(series).any() or np.isinf(series).any():
            self.logger_series.warning("Dizide NaN veya Inf değerleri var, işlem durduruldu.")
            return True
        return False

    def calculate_dtw_distance(self, series1, series2):
        """Dinamik Zaman Sargılama (DTW) mesafesini hesapla."""
        distance_dtw, path = fastdtw(series1, series2, dist=euclidean)
        return distance_dtw

    def calculate_pearson_correlation(self, series1, series2):
        """Pearson Korelasyon Katsayısını hesapla."""
        # Pearson korelasyonu -1 ile 1 arasında olduğu için (corr + 1)/2 * 100 ile 0-100 aralığına getiriyoruz
        pearson_corr_x, _ = pearsonr(series1[:, 0], series2[:, 0])
        pearson_corr_y, _ = pearsonr(series1[:, 1], series2[:, 1])
        # Ortalama Pearson korelasyonu
        pearson_corr = (pearson_corr_x + pearson_corr_y) / 2
        # Normalize et
        pearson_normalized = (pearson_corr + 1) / 2 * 100
        return pearson_normalized

    def append_to_json(self, json_path, data):
        """
        Belirtilen JSON dosyasına veri ekler. Dosya yoksa ve dizinler mevcut değilse oluşturur.
        Bu metod thread-safe bir şekilde çalışır.
        """
        try:
            with self.json_lock:
                # JSON dosyasının bulunduğu dizinin var olup olmadığını kontrol et, yoksa oluştur
                directory = os.path.dirname(json_path)
                if not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                    self.logger_series.info(f"Created directory: {directory}")

                # JSON dosyasını okuyun, varsa
                if os.path.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        try:
                            existing_data = json.load(f)
                        except json.JSONDecodeError:
                            existing_data = []
                            self.logger_series.warning(f"JSON dosyası boş veya hatalı: {json_path}")
                else:
                    existing_data = []
                    self.logger_series.info(f"Creating new JSON file: {json_path}")

                # Yeni veriyi ekleyin
                existing_data.append(data)

                # JSON dosyasını güncelle
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, ensure_ascii=False, indent=4)
                    self.logger_series.info(f"Added new data to JSON file: {json_path}")
        except Exception as e:
            self.logger_series.error(f"Error writing to JSON file {json_path}: {e}")

    def match_shapes(self, previous_series, current_series, usecase, checkName):
        """
        İki seriyi karşılaştırır ve benzerlik skoruna göre sonuç döner.
        Aynı veya farklı seriler için ilgili JSON dosyalarına veri kaydeder.
        """
        try:
            # Orijinal serileri kopyala
            previous_series1 = previous_series.copy()
            current_series1 = current_series.copy()

            # Serileri normalize et
            previous_series = self.normalize_shape(previous_series)
            current_series = self.normalize_shape(current_series)

            # NaN veya Inf kontrolü
            if self.check_for_nan_or_inf(previous_series) or self.check_for_nan_or_inf(current_series):
                self.logger_series.warning("Serilerde NaN veya Inf değerleri var.")
                return 2  # Hata durumu

            # Dinamik Zaman Sargılama (DTW) mesafesini hesapla
            dtw_distance = self.calculate_dtw_distance(previous_series, current_series)
            self.logger_series.info(f"DTW Distance: {dtw_distance}")

            # Prokrustes analizi ile şekil karşılaştırması yap
            mtx1, mtx2, disparity = procrustes(previous_series, current_series)
            self.logger_series.info(f"Procrustes Disparity: {disparity}")

            # Pearson Korelasyon Katsayısını hesapla
            pearson_corr = self.calculate_pearson_correlation(previous_series, current_series)
            self.logger_series.info(f"Pearson Correlation: {pearson_corr}")

            # Normalize et
            similarity_dtw = max(0, 100 - (dtw_distance / self.tolerance_dtw) * 100)
            similarity_procrustes = max(0, 100 - (disparity / self.tolerance_procrustes) * 100)
            similarity_pearson = pearson_corr  # Zaten 0-100 arasında

            self.logger_series.info(f"Similarity DTW: {similarity_dtw}")
            self.logger_series.info(f"Similarity Procrustes: {similarity_procrustes}")
            self.logger_series.info(f"Similarity Pearson: {similarity_pearson}")
            self.logger_series.info(f"Tolerance DTW: {self.tolerance_dtw}")

            # Combined Score Hesaplama (%40 DTW, %30 Prokrustes, %30 Pearson)
            combined_score = (0.4 * similarity_dtw) + (0.3 * similarity_procrustes) + (0.3 * similarity_pearson)
            self.logger_series.info(f"Combined Score: {combined_score}")

            # Görsel adı oluştur
            image_name = f"{datetime.now().strftime('%Y-%m-%d_%H_%M_%S_%f')}_{checkName}.png"
            current_date = datetime.now().strftime('%Y-%m-%d')
            base_path = os.path.join(self.current_dir, "logs", "series_chart", current_date, usecase)

            # JSON veri yapısı oluştur
            json_data = {
                "imageName": image_name,
                "previous_series": previous_series1.tolist(),  # NumPy dizilerini JSON uyumlu hale getirmek için listeye çevir
                "current_series": current_series1.tolist(),
                "timestamp": datetime.now().isoformat(),
                "similarity_score": combined_score
            }

            if combined_score >= 50:  # 50 ve üzeri benzer olarak kabul ediyoruz
                # Kaydetme yolu
                save_path = os.path.join(base_path, "ayni_seri", image_name)
                self.plot_shapes(previous_series, current_series, combined_score, save_path)

                # JSON yolunu belirle
                json_path = os.path.join(base_path, "dataAyni.json")

                # JSON dosyasına veri ekle
                self.append_to_json(json_path, json_data)

                return 0  # Aynı seri
            else:
                # Kaydetme yolu
                save_path = os.path.join(base_path, "farkli_seri", image_name)
                self.plot_shapes_save(previous_series, current_series, combined_score, save_path)

                # JSON yolunu belirle
                json_path = os.path.join(base_path, "dataFarkli.json")

                # JSON dosyasına veri ekle
                self.append_to_json(json_path, json_data)

                return 1  # Farklı seri
        except Exception as e:
            self.logger_series.error(f"Error in match_shapes: {e}")
            return -1  # Hata durumu

    def plot_shapes(self, previous_series, current_series, combined_score, save_path):
        """Şekilleri çizme ve dosyaya kaydetme fonksiyonu"""
        try:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))

            # İlk şekil (previous_series)
            ax[0].plot(previous_series[:, 0], previous_series[:, 1], 'bo-', label='Previous')
            ax[0].set_title('Previous Series')
            ax[0].legend()

            # İkinci şekil (current_series)
            ax[1].plot(current_series[:, 0], current_series[:, 1], 'ro-', label='Current')
            ax[1].set_title('Current Series')
            ax[1].legend()

            # Harmanlanmış skoru grafik üzerine ekleme
            fig.suptitle(f'Harmanlanmış Skor: {combined_score:.2f} / 100', fontsize=16)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(save_path)
            plt.close()
            self.logger_series.info(f"Saved plot to {save_path}")
        except Exception as e:
            self.logger_series.error(f"Error plotting shapes: {e}")

    def plot_shapes_save(self, previous_series, current_series, combined_score, save_path):
        """Farklı serileri çizme ve kaydetme fonksiyonu"""
        self.plot_shapes(previous_series, current_series, combined_score, save_path)

    def setup_logger_series(self):
        """Log kayıt methodu."""
        self.logger_series = logging.getLogger(self.__class__.__name__)
        log_dir = os.path.join(self.current_dir, "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{self.__class__.__name__}.log")
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        if not self.logger_series.handlers:
            self.logger_series.addHandler(handler)
        self.logger_series.setLevel(logging.INFO)

def store_item_as_cam_json(item_data, cam_folder, cam_file_name):
    """
    Verilen 'item_data' içeriğini,
    cam_folder/cam_file_name olarak kaydeder.
    """
    try:
        os.makedirs(cam_folder, exist_ok=True)
        file_path = os.path.join(cam_folder, cam_file_name)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(item_data, f, ensure_ascii=False, indent=4)
        print(f"Successfully saved {file_path}")
    except Exception as e:
        print(f"Error saving JSON to {file_path}: {e}")

def separate_cams(json_path):
    """
    JSON dosyasındaki öğeleri kontrol eder ve sadece 'imageName' içinde 'check_1_2' 
    içerenleri işleyerek cam klasörlerine ayırır.
    """
    # JSON dosyasını yükle
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded JSON file with {len(data)} items.")
    except Exception as e:
        print(f"JSON dosyası yüklenemedi: {e}")
        return

    # Karşılaştırma aracı (ShapeMatcher) örneği oluştur
    shape_matcher = ShapeMatcher()

    # Eğer hiç öğe yoksa çık
    if not data:
        print("JSON içerisinde öğe bulunamadı.")
        return

    # Başlangıç ayarları
    cam_index = 1          # "cam1" ile başlayacağız
    cam_file_index = 1     # cam klasörü içindeki dosya numarası
    current_cam_folder = os.path.join("logx/data", "3001json", "intermac4", f"cam{cam_index}")

    last_item = None
    processed_items = 0
    saved_items = 0

    for idx, item in enumerate(data):
        image_name = item.get("imageName", "")
        if "check_1_2" not in image_name:
            print(f"Item {idx} skipped: 'check_1_2' not in imageName ({image_name})")
            continue  # Şartı sağlamıyorsa atla

        processed_items += 1
        print(f"Processing item {idx} with imageName: {image_name}")

        if last_item is None:
            # İlk uygun öğe
            last_item = item
            cam_file_name = f"{cam_file_index}.cam.json"
            store_item_as_cam_json(last_item, current_cam_folder, cam_file_name)
            cam_file_index += 1
            saved_items += 1
            print(f"Saved first item to {os.path.join(current_cam_folder, cam_file_name)}")
            continue

        # Compare with last_item
        try:
            current_series = np.array(last_item["current_series"])
            next_series = np.array(item["current_series"])
        except KeyError as e:
            print(f"Item {idx} missing 'current_series' field: {e}")
            continue
        except Exception as e:
            print(f"Error converting 'current_series' to array for item {idx}: {e}")
            continue

        result = shape_matcher.match_shapes(
            previous_series=current_series,
            current_series=next_series,
            usecase="intermac4",
            checkName=f"compare_{idx}"
        )

        if result == 0:
            # AYNI => mevcut cam klasöründe devam ederiz
            cam_file_name = f"{cam_file_index}.cam.json"
            store_item_as_cam_json(item, current_cam_folder, cam_file_name)
            cam_file_index += 1
            saved_items += 1
            print(f"Saved same series to {os.path.join(current_cam_folder, cam_file_name)}")
        elif result == 1:
            # FARKLI => yeni cam klasörüne geçeriz, index'i artırırız
            cam_index += 1
            cam_file_index = 1
            current_cam_folder = os.path.join("logx/data", "3001json", "intermac4", f"cam{cam_index}")
            cam_file_name = f"{cam_file_index}.cam.json"
            store_item_as_cam_json(item, current_cam_folder, cam_file_name)
            cam_file_index += 1
            saved_items += 1
            print(f"Saved different series to {os.path.join(current_cam_folder, cam_file_name)}")
        else:
            print(f"Seri karşılaştırmasında hata oluştu: {image_name}")

        # Son kaydedilen öğeyi güncelle
        last_item = item

    print(f"Processed {processed_items} items with 'check_1_2' in imageName.")
    print(f"Saved {saved_items} items into cam folders.")

def main():
    # Örnek kullanım:
    # Elinizdeki JSON dosya yolunu girin
    input_json_path = "E:/Proje repo genel/3001-works/logx/3001json/intermac4/dataAyni.json"

    # Bu fonksiyon, JSON içindeki 'check1_2' içeren tüm öğeleri
    # logx/2101json/intermac4/cam1/1.cam.json, cam1/2.cam.json, cam2/1.cam.json vb. şeklinde
    # ayrı klasörlere kaydedecektir.
    separate_cams(input_json_path)

if __name__ == "__main__":
    main()

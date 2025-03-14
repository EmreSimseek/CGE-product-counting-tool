import json
from datetime import datetime
import matplotlib.pyplot as plt
import os
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.spatial import procrustes
from scipy.interpolate import interp1d
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
            print("Dizide NaN veya Inf değerleri var, işlem durduruldu.")
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

            # JSON dosyasını güncelleyin
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=4)
                self.logger_series.info(f"Added new data to JSON file: {json_path}")
        except Exception as e:
            self.logger_series.error(f"Error writing to JSON file {json_path}: {e}")

    def match_shapes(self, previous_series, current_series, usecase, checkName):
        """
        İki seriyi karşılaştırır ve benzerlik skorunu döndürür.
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
                return 0.0  # Or any default value

            # Dinamik Zaman Sargılama (DTW) mesafesini hesapla
            dtw_distance = self.calculate_dtw_distance(previous_series, current_series)
            
            # Prokrustes analizi ile şekil karşılaştırması yap
            mtx1, mtx2, disparity = procrustes(previous_series, current_series)

            # Pearson Korelasyon Katsayısını hesapla
            pearson_corr = self.calculate_pearson_correlation(previous_series, current_series)
            print("disparity",disparity)
            # Normalize et
            similarity_dtw = max(0, 100 - (dtw_distance / self.tolerance_dtw) * 100)
            similarity_procrustes = max(0, 100 - (disparity / self.tolerance_procrustes) * 100)
            similarity_pearson = pearson_corr  # Zaten 0-100 arasında

            # Combined Score Hesaplama (%40 DTW, %30 Prokrustes, %30 Pearson)
            combined_score = (0.4 * similarity_dtw) + (0.3 * similarity_procrustes) + (0.3 * similarity_pearson)

            # Görsel adı oluştur
            image_name = f"{datetime.now().strftime('%Y-%m-%d_%H_%M_%S_%f')}_{checkName}.png"
            #if dtw_distance >3000:
            print(f"dtw distance:",dtw_distance)
            print(f"procrustes :",disparity)
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
                json_path = os.path.join(base_path, "dataAynis.json")

                # JSON dosyasına veri ekle
                self.append_to_json(json_path, json_data)
            else:
                # Kaydetme yolu
                save_path = os.path.join(base_path, "farkli_seri", image_name)
                self.plot_shapes_save(previous_series, current_series, combined_score, save_path)

                # JSON yolunu belirle
                json_path = os.path.join(base_path, "dataFarklis.json")

                # JSON dosyasına veri ekle
                self.append_to_json(json_path, json_data)

            return combined_score  # Return the similarity score
        except Exception as e:
            self.logger_series.error(f"Error Checking from Series Detect {e}")
            return 0.0  # Or any default value

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
        except Exception as e:
            self.logger_series.error(f"Error plotting shapes: {e}")

    def plot_shapes_save(self, previous_series, current_series, combined_score, save_path):
        """Farklı serileri çizme ve kaydetme fonksiyonu"""
        self.plot_shapes(previous_series, current_series, combined_score, save_path)

    def setup_logger_series(self):
        """Log kayıt methodu."""
        self.logger_series = logging.getLogger(self.__class__.__name__)
        log_file = os.path.join(self.current_dir, "logs", f"{self.__class__.__name__}.log")
        if not os.path.exists(os.path.dirname(log_file)):
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        if not self.logger_series.handlers:
            self.logger_series.addHandler(handler)
        self.logger_series.setLevel(logging.INFO)

class ShapeViewerGUI:
    def __init__(self, master, data):
        self.master = master
        self.master.title("Shape Viewer")
        self.data = data
        self.current_index = 0

        # Klavye kısayollarını bağlama
        self.master.bind("<n>", self.key_press)  # 'n' tuşu için
        self.master.bind("<b>", self.key_press)  # 'b' tuşu için

        # Frame for plot
        self.plot_frame = ttk.Frame(master)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        # Frame for buttons
        self.button_frame = ttk.Frame(master)
        self.button_frame.pack(fill=tk.X)

        self.back_button = ttk.Button(self.button_frame, text="Back", command=self.show_previous)
        self.back_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.next_button = ttk.Button(self.button_frame, text="Next", command=self.show_next)
        self.next_button.pack(side=tk.RIGHT, padx=10, pady=10)

        # Initialize plot
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Display the first entry
        self.display_entry()

    def display_entry(self):
        entry = self.data[self.current_index]
        previous = np.array(entry["previous_series"])
        current = np.array(entry["current_series"])
        timestamp = entry["timestamp"]
        similarity_score = entry["similarity_score"]

        self.ax.clear()

        # Previous Series
        self.ax.plot(previous[:, 0], previous[:, 1], 'bo-', label='Previous Series')

        # Current Series
        self.ax.plot(current[:, 0], current[:, 1], 'ro-', label='Current Series')

        # Başlık ve Etiketler
        self.ax.set_title(f"Image: {entry['imageName']}")
        self.ax.set_xlabel("X Koordinatı")
        self.ax.set_ylabel("Y Koordinatı")
        self.ax.legend()

        # Saat ve Benzerlik Skorunu Alt Kısımda Göstermek
        plot_text = (
            f"Timestamp: {datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Similarity Score: {similarity_score:.2f}"
        )

        # Metni grafiğin altına ekleme
        self.fig.subplots_adjust(bottom=0.3)  # Alt kısım için daha fazla boşluk ayır
        self.fig.text(
            0.5, 0.05, plot_text,
            horizontalalignment='center',
            verticalalignment='center',
            fontsize=10,
            bbox=dict(facecolor='lightgrey', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.5)
        )

        self.canvas.draw()

    def show_next(self):
        if self.current_index < len(self.data) - 1:
            self.current_index += 1
            self.display_entry()
        else:
            messagebox.showinfo("Bilgi", "Son veriye ulaşıldı.")

    def show_previous(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.display_entry()
        else:
            messagebox.showinfo("Bilgi", "İlk veridesiniz.")
        
    def key_press(self, event):
        """Tuşa basıldığında çalışır."""
        if event.keysym == 'n':  # 'n' tuşu için
            self.show_next()
        elif event.keysym == 'b':  # 'b' tuşu için
            self.show_previous()

def load_json(file_path):
    """JSON dosyasını yükler."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def filter_and_sort_data(data_list, keyword="check_1_2.png"):
    """Verileri filtreler ve timestamp'e göre sıralar."""
    # Filtreleme
    filtered_data = [
        entry for entry in data_list
        if keyword in entry.get("imageName", "")
    ]

    # Tarihe göre sıralama
    sorted_data = sorted(
        filtered_data,
        key=lambda x: datetime.fromisoformat(x["timestamp"])
    )

    return sorted_data

def process_data_with_shape_matcher(data, shape_matcher):
    """
    ShapeMatcher kullanarak verileri işler ve similarity_score'u günceller.
    """
    processed_data = []
    for entry in data:
        # imageName'den usecase ve checkName'ı çıkarın
        # Örneğin, "2025-01-21_08_22_30_175209_check_1_2.png"
        # usecase'ı belirlemek için uygun bir yöntem kullanın. Örneğin, "check" kelimesinden önceki kısmı usecase olarak alabilirsiniz.
        # Bu örnekte, basitçe "check_1_2" olarak alıyoruz.
        image_name = entry.get("imageName", "")
        check_name = image_name.split("_")[-1].replace(".png", "")  # "check_1_2"

        # usecase'ı belirlemek için image_name'den uygun bir bölüm alın
        # Örneğin, "aynı" veya "farklı"
        if "aynı" in image_name.lower():
            usecase = "aynı_seri"
        elif "farklı" in image_name.lower():
            usecase = "farkli_seri"
        else:
            usecase = "unknown_seri"

        # ShapeMatcher ile benzerlik skorunu hesapla
        similarity_score = shape_matcher.match_shapes(
            previous_series=entry["previous_series"],
            current_series=entry["current_series"],
            usecase=usecase,
            checkName=check_name
        )

        # similarity_score'u veri entry'sine ekle veya güncelle
        entry["similarity_score"] = similarity_score

        # processed_data listesine ekle
        processed_data.append(entry)

    return processed_data

def main():
    # JSON dosya yollarınızı buraya ekleyin
    json_file_1 = "/home/emre/Masaüstü/my_development/2101json/intermac4/dataAyni.json"  # Örneğin: "data1.json"
    json_file_2 = "/home/emre/Masaüstü/my_development/2101json/intermac4/dataFarkli.json" # Örneğin: "data2.json"

    # JSON verilerini yükle
    try:
        data1 = load_json(json_file_1)
        data2 = load_json(json_file_2)
    except FileNotFoundError as e:
        messagebox.showerror("Hata", f"Dosya bulunamadı: {e}")
        return
    except json.JSONDecodeError as e:
        messagebox.showerror("Hata", f"JSON decode hatası: {e}")
        return

    # Verileri birleştir
    combined_data = data1 + data2

    # Verileri filtrele ve sırala
    sorted_filtered_data = filter_and_sort_data(combined_data, keyword="check_1_2.png")

    # Eğer filtrelenmiş veri yoksa, kullanıcıyı bilgilendir
    if not sorted_filtered_data:
        messagebox.showinfo("Bilgi", "imageName içinde 'check_1_2.png' geçen hiçbir giriş bulunamadı.")
        return

    # ShapeMatcher örneği oluştur
    shape_matcher = ShapeMatcher()

    # Verileri ShapeMatcher ile işle
    processed_data = process_data_with_shape_matcher(sorted_filtered_data, shape_matcher)

    # Tkinter penceresini başlat
    root = tk.Tk()
    app = ShapeViewerGUI(root, processed_data)
    root.mainloop()

if __name__ == "__main__":
    main()

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


class ShapeViewerGUI:
    def __init__(self, master, data):
        self.master = master
        self.master.title("Shape Viewer")
        self.data = data
        self.current_index = 0
        self.display_mode = 1  # 1: Overlaid, 2: Side by Side
        self.normalize_mode = False  # Normalize modu: False -> Orijinal, True -> Normalize edilmiş

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
        self.next_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.mode_button = ttk.Button(self.button_frame, text="Mod 1", command=self.toggle_mode)
        self.mode_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.normalize_button = ttk.Button(
            self.button_frame,
            text="Normalize: OFF",
            command=self.toggle_normalize
        )
        self.normalize_button.pack(side=tk.LEFT, padx=10, pady=10)

        # Initialize plot
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Display the first entry
        self.display_entry()

    def toggle_mode(self):
        """Modu değiştirir ve grafiği günceller."""
        self.display_mode = 2 if self.display_mode == 1 else 1
        self.mode_button.config(text=f"Mod {self.display_mode}")
        self.display_entry()

    def toggle_normalize(self):
        """Normalize modunu değiştirir ve grafiği günceller."""
        self.normalize_mode = not self.normalize_mode
        self.normalize_button.config(
            text=f"Normalize: {'ON' if self.normalize_mode else 'OFF'}"
        )
        self.display_entry()

    def normalize_shape(self, series, num_points=100):
        """Nokta sayısını normalize et."""
        num_points_original = len(series)
        if num_points_original == num_points:
            return series  # Eğer zaten doğru sayıda nokta varsa, işlem yapma

        # Orijinal ve hedef indeksler oluştur
        original_indices = np.linspace(0, 1, num=num_points_original)
        target_indices = np.linspace(0, 1, num=num_points)

        # Her iki eksen için ara değerleme yap
        interpolator = interp1d(original_indices, series, axis=0, kind='cubic', fill_value="extrapolate")
        normalized_series = interpolator(target_indices)

        return normalized_series

    def display_entry(self):
        entry = self.data[self.current_index]
        current = np.array(entry["previous_series"])  ##current ve previosu yer değişti app videos parametreleri yüzünden
        previous = np.array(entry["current_series"])

        # Normalize edilmiş serileri kullan veya orijinal serileri kullan
        if self.normalize_mode:
            current = self.normalize_shape(current)
            previous = self.normalize_shape(previous)

        timestamp = entry["timestamp"]
        similarity_score = entry["similarity_score"]

        self.ax.clear()

        # Grafik yapılandırmasını sıfırla
        self.fig.clf()

        if self.display_mode == 1:
            # Mod 1: Üst üste
            self.ax = self.fig.add_subplot(111)
            self.ax.plot(previous[:, 0], previous[:, 1], 'bo-', label='Previous Series')
            self.ax.plot(current[:, 0], current[:, 1], 'ro-', label='Current Series')
            self.ax.set_title(f"Image: {entry['imageName']}")
            self.ax.set_xlabel("X Koordinatı")
            self.ax.set_ylabel("Y Koordinatı")
            self.ax.legend()
        elif self.display_mode == 2:
            # Mod 2: Yan yana
            self.ax_prev = self.fig.add_subplot(1, 2, 1)
            self.ax_prev.plot(previous[:, 0], previous[:, 1], 'bo-', label='Previous Series')
            self.ax_prev.set_title('Previous Series')
            self.ax_prev.set_xlabel("X Koordinatı")
            self.ax_prev.set_ylabel("Y Koordinatı")
            self.ax_prev.legend()

            self.ax_curr = self.fig.add_subplot(1, 2, 2)
            self.ax_curr.plot(current[:, 0], current[:, 1], 'ro-', label='Current Series')
            self.ax_curr.set_title('Current Series')
            self.ax_curr.set_xlabel("X Koordinatı")
            self.ax_curr.set_ylabel("Y Koordinatı")
            self.ax_curr.legend()

            self.fig.suptitle(f"Image: {entry['imageName']}", fontsize=16)

        # Saat ve Benzerlik Skorunu Eklemek
        plot_text = (
            f"Timestamp: {datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Similarity Score: {similarity_score:.2f}"
        )
        # Metni grafiğin sağ alt köşesine ekleme
        self.fig.subplots_adjust(bottom=0.2)
        self.fig.text(
            0.95, 0.02, plot_text,
            horizontalalignment='right',
            verticalalignment='bottom',
            fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8)
        )

        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
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

def main():
    # JSON dosya yollarınızı buraya ekleyin # Bir gün boyunca kesilen TÜM camların aynı - veya farklı olan json verisinin yeni parametrelerle ayrılmış hali 
    json_file_1 = "/home/emre/Masaüstü/my_development/logs/series_chart/2025-01-22/intermac4-farkli-check-1.5-10000-100/dataAyni.json"  
    json_file_2 = "/home/emre/Masaüstü/my_development/logs/series_chart/2025-01-22/intermac4-farkli-check-1.5-10000-100/dataFarkli.json" 
                                                                                                                                            
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

    # Tkinter penceresini başlat
    root = tk.Tk()
    app = ShapeViewerGUI(root, sorted_filtered_data)
    root.mainloop()

if __name__ == "__main__":
    main()

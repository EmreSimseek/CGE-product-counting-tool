import json
from datetime import datetime
import matplotlib.pyplot as plt
import os
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.interpolate import interp1d
import logging

class ShapeViewerGUI:
    def __init__(self, master, logx_folder):
        self.master = master
        self.master.title("Shape Viewer")
        self.current_index = 0
        self.display_mode = 1  # 1: Üst üste, 2: Yan yana
        self.normalize_mode = False  # Normalize modu: False -> Orijinal, True -> Normalize edilmiş
        self.data = []
        self.logx_folder = logx_folder

        # Logger
        self.setup_logger()

        # Menü
        self.setup_menu()

        # Frame for plot
        self.plot_frame = ttk.Frame(master)
        self.plot_frame.pack(fill=tk.BOTH, expand=True)

        # Frame for buttons
        self.button_frame = ttk.Frame(master)
        self.button_frame.pack(fill=tk.X)

        self.back_button = ttk.Button(self.button_frame, text="Geri", command=self.show_previous)
        self.back_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.next_button = ttk.Button(self.button_frame, text="İleri", command=self.show_next)
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

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Veri Yükleniyor...")
        self.status_bar = ttk.Label(master, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(fill=tk.X, side=tk.BOTTOM)

        # Key bindings
        self.master.bind('<n>', self.on_key_press)
        self.master.bind('<b>', self.on_key_press)

        # Yükleme işlemini başlat
        self.load_data()

    def setup_logger(self):
        """Logger yapılandırması."""
        self.logger = logging.getLogger("ShapeViewerGUI")
        self.logger.setLevel(logging.INFO)
        log_dir = os.path.join(os.getcwd(), "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "ShapeViewerGUI.log")
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        if not self.logger.handlers:
            self.logger.addHandler(handler)

    def setup_menu(self):
        """Menü yapılandırması."""
        self.menu = tk.Menu(self.master)
        self.master.config(menu=self.menu)
        file_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="Dosya", menu=file_menu)
        file_menu.add_command(label="Çıkış", command=self.master.quit)

    def load_data(self):
        """Belirtilen logx klasöründeki tüm JSON dosyalarını yükler."""
        self.logger.info(f"Logx klasörü: {self.logx_folder}")
        if not os.path.exists(self.logx_folder):
            messagebox.showerror("Hata", f"Belirtilen klasör mevcut değil: {self.logx_folder}")
            self.logger.error(f"Belirtilen klasör mevcut değil: {self.logx_folder}")
            self.status_var.set("Hata: Klasör bulunamadı.")
            return

        # Belirtilen klasörün adını 'usecase' olarak al
        usecase = os.path.basename(os.path.normpath(self.logx_folder))
        self.logger.info(f"Usecase olarak belirlendi: {usecase}")

        # Tüm intermac ve cam klasörlerini tarayın
        for root, dirs, files in os.walk(self.logx_folder):
            for file in files:
                if file.endswith(".json"):
                    json_path = os.path.join(root, file)
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            entry = json.load(f)
                            # Klasör adlarını ekle (intermacX/camY)
                            relative_path = os.path.relpath(root, self.logx_folder)
                            path_parts = relative_path.split(os.sep)
                            if len(path_parts) >=1 and path_parts[0] != '.':
                                cam = path_parts[-1]      # camY
                            else:
                                cam = "Unknown Cam"
                            entry['usecase'] = usecase
                            entry['cam'] = cam
                            self.data.append(entry)
                        self.logger.info(f"Yüklendi: {json_path} | Usecase: {usecase}, Cam: {cam}")
                    except Exception as e:
                        self.logger.error(f"Yükleme hatası {json_path}: {e}")

        if not self.data:
            messagebox.showinfo("Bilgi", "Seçilen klasörde JSON dosyası bulunamadı.")
            self.logger.warning("Seçilen klasörde JSON dosyası bulunamadı.")
            self.status_var.set("Hata: JSON dosyası bulunamadı.")
            return

        # Verileri sıralama (isteğe bağlı, örneğin timestamp'e göre)
        try:
            self.data.sort(key=lambda x: datetime.fromisoformat(x.get("timestamp", "1970-01-01T00:00:00")))
        except Exception as e:
            self.logger.error(f"Sıralama hatası: {e}")

        self.current_index = 0
        self.display_entry()
        self.logger.info(f"Toplam {len(self.data)} giriş yüklendi.")

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
        try:
            interpolator = interp1d(original_indices, series, axis=0, kind='cubic', fill_value="extrapolate")
            normalized_series = interpolator(target_indices)
            return normalized_series
        except Exception as e:
            self.logger.error(f"Normalize hatası: {e}")
            return series  # Hata durumunda orijinal seriyi döndür

    def display_entry(self):
        if not self.data:
            self.logger.warning("Gösterilecek veri yok.")
            return

        entry = self.data[self.current_index]
        previous = np.array(entry.get("previous_series", []))
        current = np.array(entry.get("current_series", []))

        # Normalize edilmiş serileri kullan veya orijinal serileri kullan
        if self.normalize_mode and len(previous) > 0 and len(current) > 0:
            previous = self.normalize_shape(previous)
            current = self.normalize_shape(current)

        timestamp = entry.get("timestamp", "Unknown Timestamp")
        similarity_score = entry.get("similarity_score", 0.0)
        usecase = entry.get('usecase', 'Unknown Usecase')
        cam = entry.get('cam', 'Unknown Cam')

        self.fig.clf()

        if self.display_mode == 1:
            # Mod 1: Üst üste
            self.ax = self.fig.add_subplot(111)
            if len(previous) > 0:
                self.ax.plot(previous[:, 0], previous[:, 1], 'bo-', label='Previous Series')
            if len(current) > 0:
                self.ax.plot(current[:, 0], current[:, 1], 'ro-', label='Current Series')
            self.ax.set_title(f"Path: {usecase}/{cam}\nImage: {entry.get('imageName', '')}")
            self.ax.set_xlabel("X Koordinatı")
            self.ax.set_ylabel("Y Koordinatı")
            self.ax.legend()
        elif self.display_mode == 2:
            # Mod 2: Yan yana
            self.ax_prev = self.fig.add_subplot(1, 2, 1)
            if len(previous) > 0:
                self.ax_prev.plot(previous[:, 0], previous[:, 1], 'bo-', label='Previous Series')
            self.ax_prev.set_title('Previous Series')
            self.ax_prev.set_xlabel("X Koordinatı")
            self.ax_prev.set_ylabel("Y Koordinatı")
            self.ax_prev.legend()

            self.ax_curr = self.fig.add_subplot(1, 2, 2)
            if len(current) > 0:
                self.ax_curr.plot(current[:, 0], current[:, 1], 'ro-', label='Current Series')
            self.ax_curr.set_title('Current Series')
            self.ax_curr.set_xlabel("X Koordinatı")
            self.ax_curr.set_ylabel("Y Koordinatı")
            self.ax_curr.legend()

            self.fig.suptitle(f"Path: {usecase}/{cam}\nImage: {entry.get('imageName', '')}", fontsize=16)

        # Saat ve Benzerlik Skorunu Eklemek
        try:
            plot_text = (
                f"Timestamp: {datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Similarity Score: {similarity_score:.2f}"
            )
        except Exception as e:
            plot_text = f"Timestamp: {timestamp}\nSimilarity Score: {similarity_score:.2f}"
            self.logger.error(f"Timestamp format hatası: {e}")

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

        # Güncel durumu güncelle
        self.status_var.set(f"Path: {usecase}/{cam} | Entry: {self.current_index + 1}/{len(self.data)}")

    def show_next(self):
        if self.current_index < len(self.data) - 1:
            self.current_index += 1
            self.display_entry()
        else:
            messagebox.showinfo("Bilgi", "Son veriye ulaşıldı.")
            self.logger.info("Son veriye ulaşıldı.")

    def show_previous(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.display_entry()
        else:
            messagebox.showinfo("Bilgi", "İlk veridesiniz.")
            self.logger.info("İlk veridesiniz.")

    def on_key_press(self, event):
        """Tuşa basıldığında çalışır."""
        if event.char == 'n':  # 'n' tuşu için
            self.show_next()
        elif event.char == 'b':  # 'b' tuşu için
            self.show_previous()

def main():
    # Klasör yolunu burada belirleyin
    logx_folder = "E:/Proje repo genel/3001-works/logx/data/3001json/intermac2"  # Kendi klasör yolunuzu buraya girin

    # Klasörün var olup olmadığını kontrol edin
    if not os.path.exists(logx_folder):
        print(f"Belirtilen klasör mevcut değil: {logx_folder}")
        return

    # Tkinter penceresini başlat
    root = tk.Tk()
    app = ShapeViewerGUI(root, logx_folder)
    root.mainloop()

if __name__ == "__main__":
    main()

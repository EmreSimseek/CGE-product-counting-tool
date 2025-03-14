import os
import json
import numpy as np
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import threading
from scipy.spatial.distance import euclidean
from scipy.spatial import procrustes
from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from fastdtw import fastdtw
from collections import defaultdict
import cv2  # OpenCV, pip install opencv-python
from math import isnan, isinf
from shapely.geometry import MultiPoint, Polygon
###############################################################################
# 1) Güncellenmiş ShapeMatcher Sınıfı
###############################################################################

class ShapeMatcher:
    """
    Bu sınıf, DTW, Procrustes, Pearson gibi metriklere ek olarak,
    iki nokta dizisinin alan ve şekil benzerliğini (konveks kabuk & Hu momentları) hesaplar.
    Ayrıca, outlier noktalarını tespit edip kaydeder ve sonuçları plot içine metin olarak ekler.
    """
    def __init__(self, tolerance_dtw=3000, tolerance_procrustes=0.06, tolerance_pearson=100, num_points=100):
        self.current_dir = os.getcwd()
        self.setup_logger_series()
        self.tolerance_dtw = tolerance_dtw
        self.tolerance_procrustes = tolerance_procrustes
        self.tolerance_pearson = tolerance_pearson
        self.num_points = num_points  
        self.json_lock = threading.Lock()
        self.prev_outliers = []
        self.curr_outliers = []

    def setup_logger_series(self):
        self.logger_series = logging.getLogger(self.__class__.__name__)
        logs_dir = os.path.join(self.current_dir, "logs")
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir, exist_ok=True)
        log_file = os.path.join(logs_dir, f"{self.__class__.__name__}.log")
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
        handler.setFormatter(formatter)
        if not self.logger_series.handlers:
            self.logger_series.addHandler(handler)
        self.logger_series.setLevel(logging.INFO)

    def normalize_shape(self, series):
        num_points_original = len(series)
        if num_points_original == self.num_points:
            return series
        original_indices = np.linspace(0, 1, num=num_points_original)
        target_indices = np.linspace(0, 1, num=self.num_points)
        interpolator = interp1d(original_indices, series, axis=0, kind='cubic', fill_value="extrapolate")
        normalized_series = interpolator(target_indices)
        return normalized_series

    def check_for_nan_or_inf(self, series):
        if np.isnan(series).any() or np.isinf(series).any():
            print("Dizide NaN veya Inf değerleri var, işlem durduruldu.")
            self.logger_series.warning("Dizide NaN veya Inf değerleri var, işlem durduruldu.")
            return True
        return False

    def calculate_dtw_distance(self, series1, series2):
        distance_dtw, _ = fastdtw(series1, series2, dist=euclidean)
        return distance_dtw

    def calculate_pearson_correlation(self, series1, series2):
        pearson_corr_x, _ = pearsonr(series1[:, 0], series2[:, 0])
        pearson_corr_y, _ = pearsonr(series1[:, 1], series2[:, 1])
        pearson_corr = (pearson_corr_x + pearson_corr_y) / 2
        pearson_normalized = (pearson_corr + 1) / 2 * 100
        return pearson_normalized

    def identify_outliers(self, series, distance_factor=3, small_gap_threshold=3,
                      head_gap_threshold=5, tail_gap_threshold=5):
        """
        İki ardışık nokta arasındaki mesafeleri hesaplayıp,
        medyanın distance_factor katından büyük olanları outlier olarak işaretler.
        
        Ekstra adımlar:
        1. Bir noktanın hem sol hem sağ komşusu outlier ise, o nokta da outlier kabul edilir.
        2. Outlier blokları arasındaki boşluk küçükse (small_gap_threshold veya daha az),
            aradaki noktalar da outlier olarak işaretlenir.
        3. Serinin başında:
            Eğer ilk outlier belirlenene kadar gelen valid nokta sayısı (başlangıç bloğu)
            head_gap_threshold veya daha az ise, bu bloktaki tüm noktalar outlier olarak işaretlenir.
        4. Serinin sonunda:
            Eğer son outlier belirlendikten sonra gelen valid nokta sayısı tail_gap_threshold veya 
            daha az ise, bu bloktaki tüm noktalar outlier olarak işaretlenir.
        
        Parametreler:
        series: (N, D) boyutunda numpy dizisi (N nokta, D boyut).
        distance_factor: Mesafe eşik değerini belirleyen faktör (varsayılan: 3).
        small_gap_threshold: İki outlier bloğu arasındaki boşluğun, küçük sayıda nokta içerip içermediğini belirler (varsayılan: 3).
        head_gap_threshold: Serinin başındaki valid blok uzunluğu; bu sayı veya daha az ise, baştaki noktalara da outlier denir (varsayılan: 3).
        tail_gap_threshold: Serinin sonundaki valid blok uzunluğu; bu sayı veya daha az ise, sondaki noktalara da outlier denir (varsayılan: 5).
        
        Returns:
        Outlier indekslerinin sıralı listesi.
        """
        n_points = len(series)
        if n_points < 2:
            return []
        
        # İki ardışık nokta arasındaki Öklid mesafelerini hesapla.
        distances = np.linalg.norm(series[1:] - series[:-1], axis=1)
        if len(distances) == 0:
            return []
        
        median_distance = np.median(distances)
        threshold_distance = median_distance * distance_factor
        
        outlier_indices = set()
        # Mesafe eşik değerini aşan ardışık noktalar: bu noktalarda her iki uç nokta outlier kabul edilir.
        outlier_distance_indices = np.where(distances > threshold_distance)[0]
        for idx in outlier_distance_indices:
            outlier_indices.add(idx)
            outlier_indices.add(idx + 1)
        
        # Adım 1: Eğer bir noktanın hem sol hem sağ komşusu outlier ise, o nokta da outlier sayılır.
        for i in range(1, n_points - 1):
            if (i - 1 in outlier_indices) and (i + 1 in outlier_indices):
                outlier_indices.add(i)
        
        # Adım 2: Outlier blokları arasındaki kısa boşlukları da outlier olarak işaretle.
        outlier_sorted = sorted(outlier_indices)
        for i in range(len(outlier_sorted) - 1):
            current_index = outlier_sorted[i]
            next_index = outlier_sorted[i + 1]
            gap_length = next_index - current_index - 1
            if gap_length > 0 and gap_length <= small_gap_threshold:
                for j in range(current_index + 1, next_index):
                    outlier_indices.add(j)
        
        # Yeni Adım 3: Serinin başındaki valid bloğun kontrolü.
        # "Serinin başından itibaren 3 nokta gittik ve outlierlar başladı" durumunda,
        # yani ilk outlier'a kadar gelen valid nokta sayısı head_gap_threshold veya daha az ise,
        # bu valid noktalar da outlier olarak işaretlenecek.
        if outlier_indices:
            first_outlier = min(outlier_indices)
            if first_outlier > 0 and first_outlier <= head_gap_threshold:
                for i in range(first_outlier):
                    outlier_indices.add(i)
        
        # Yeni Adım 4: Serinin sonundaki valid bloğun kontrolü.
        # "Listenin sonunda sondan 5 önceki eleman ve öncesinde outlier varsa",
        # yani son outlier'dan sonraki valid nokta sayısı tail_gap_threshold veya daha az ise,
        # bu valid noktalar da outlier olarak işaretlenecek.
        if outlier_indices:
            last_outlier = max(outlier_indices)
            tail_valid_count = n_points - 1 - last_outlier
            if tail_valid_count > 0 and tail_valid_count <= tail_gap_threshold:
                for i in range(last_outlier + 1, n_points):
                    outlier_indices.add(i)
        
        return sorted(outlier_indices)

    def remove_outliers_from_series(self, series, outlier_indices):
        """Belirtilen outlier indekslerini seriden çıkarır."""
        if len(outlier_indices) == 0:
            return series
        mask = np.ones(len(series), dtype=bool)
        mask[outlier_indices] = False
        return series[mask]
    # --- Yeni eklenen alan ve şekil benzerliği metotları ---
    def compute_convex_hull_area(self, points):
        """
        Nokta dizisinin konveks kabuğunu hesaplar ve alanını döndürür.
        """
        points = points.astype(np.float32)
        hull = cv2.convexHull(points)
        area = cv2.contourArea(hull)
        return area

    def area_similarity(self, points1, points2, beta=10):
        """
        İki nokta dizisinin konveks kabuk alanlarına göre alan benzerliğini (0-100) hesaplar.
        beta: Alan farkına duyarlılığı kontrol eden katsayı.
        """
        area1 = self.compute_convex_hull_area(points1)
        area2 = self.compute_convex_hull_area(points2)
        if area1 == 0 or area2 == 0:
            return 0
        ratio_diff = abs(np.log(area1 / area2))
        sim = max(0, 100 - beta * ratio_diff)
        return sim

    def compute_hu_moments(self, points):
        """
        Nokta dizisinin (konveks kabuğu üzerinden) Hu momentlarını hesaplar.
        Hu momentları, çeviri, ölçek ve rotasyona karşı invariandır.
        """
        points = points.astype(np.float32)
        hull = cv2.convexHull(points)
        moments = cv2.moments(hull)
        huMoments = cv2.HuMoments(moments)
        huMoments_log = -np.sign(huMoments) * np.log10(np.abs(huMoments) + 1e-10)
        return huMoments_log.flatten()

    def shape_similarity(self, points1, points2, gamma=50):
        """
        Hu momentları üzerinden iki nokta dizisinin şekil benzerliğini hesaplar (0-100).
        gamma: Hu moment farkını benzerliğe çevirirken kullanılan katsayı.
        """
        hu1 = self.compute_hu_moments(points1)
        hu2 = self.compute_hu_moments(points2)
        d = np.linalg.norm(hu1 - hu2)
        sim = max(0, 100 - gamma * d)
        return sim

    def iou_similarity(self, points1, points2, iou_power=3):
        """
        İki nokta dizisinin konveks kabuklarına dayalı IoU benzerliğini hesaplar ve
        0-100 arasına ölçekler. iou_power katsayısı, IoU değeri düşükse skorun daha
        hızlı düşmesini sağlar.
        """
        points1 = points1.astype(np.float32)
        points2 = points2.astype(np.float32)
        hull1 = cv2.convexHull(points1)
        hull2 = cv2.convexHull(points2)
        area1 = cv2.contourArea(hull1)
        area2 = cv2.contourArea(hull2)
        if area1 <= 0 or area2 <= 0:
            return 0
        retval, intersection = cv2.intersectConvexConvex(hull1, hull2)
        intersection_area = retval  # cv2.intersectConvexConvex'in döndürdüğü alan
        union_area = area1 + area2 - intersection_area
        if union_area == 0:
            return 0
        iou = intersection_area / union_area
        sim = 100 * (iou ** iou_power)
        return sim
    # --- Yeni metotlar sonu ---
    def compute_penalized_similarity(self,iou_value, poly1, poly2):
        # Dış alanın büyüklüğünü hesapla (union - intersection)
        intersection_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area
        non_intersection_area = union_area - intersection_area
        
        # Penalizasyon faktörü: daha büyük dış alan, daha büyük penalizasyon
        penalty_factor = 1 + 2 * (non_intersection_area / union_area)  # Dış alan arttıkça penalizasyon artar
        similarity_score = (iou_value ** 3) / penalty_factor  # IOU ile penalizasyonu birleştiriyoruz
        
            # Ölçekleme işlemi (0-100 arası)
        min_score = 0  # Minimum skor değeri
        max_score = 1  # Maximum skor değeri, burada 1 olarak belirledim
        scaled_similarity_score = (similarity_score - min_score) / (max_score - min_score) * 100

        return scaled_similarity_score
    
    def compute_iou(self,poly1, poly2):
        if not poly1 or not poly2:
            return 0
        intersection_area = poly1.intersection(poly2).area
        union_area = poly1.union(poly2).area
        if union_area == 0:
            return 0
        return intersection_area / union_area        
    def compute_convex_hull(self,points):
        if len(points) < 3:
            return None
        multipoint = MultiPoint(points)
        return multipoint.convex_hull

    def plot_shapes(self, previous_series, current_series, combined_score, similarity_dtw, similarity_procrustes, similarity_pearson, save_path, area_sim=None, shape_sim=None):
        try:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].plot(previous_series[:, 0], previous_series[:, 1], 'bo-', label='Previous')
            ax[1].plot(current_series[:, 0], current_series[:, 1], 'ro-', label='Current')
            if len(self.prev_outliers) > 0:
                ax[0].plot(previous_series[self.prev_outliers, 0], previous_series[self.prev_outliers, 1],
                           'kx', markersize=10, label='Outlier')
            if len(self.curr_outliers) > 0:
                ax[1].plot(current_series[self.curr_outliers, 0], current_series[self.curr_outliers, 1],
                           'kx', markersize=10, label='Outlier')
            ax[0].set_title('Previous Series')
            ax[1].set_title('Current Series')
            ax[0].legend()
            ax[1].legend()
            fig.suptitle(
                f"Harmanlanmış Skor: {combined_score:.2f} / 100\n"
                f"DTW Skor: {similarity_dtw:.2f}  |  Procrustes Skor: {similarity_procrustes:.2f}  |  Pearson Skor: {similarity_pearson:.2f}",
                fontsize=14
            )
            if area_sim is not None and shape_sim is not None:
                fig.text(0.5, 0.01, f"Area Similarity: {area_sim:.2f}   |   Shape Similarity: {shape_sim:.2f}",
                         ha="center", fontsize=12)
            plt.tight_layout(rect=[0, 0.05, 1, 0.95])
            plt.savefig(save_path)
            plt.close()
        except Exception as e:
            self.logger_series.error(f"Error plotting shapes: {e}")

    def plot_shapes_save(self, previous_series, current_series, combined_score, similarity_dtw, similarity_procrustes, similarity_pearson, save_path, area_sim=None, shape_sim=None):
        self.plot_shapes(previous_series, current_series, combined_score, similarity_dtw, similarity_procrustes, similarity_pearson, save_path, area_sim, shape_sim)

    def match_shapes(self, previous_series, current_series, usecase=None, checkName=None, return_outliers=False):
        """
        İki nokta dizisini karşılaştırır, metrikleri hesaplar.
        Artı; alan ve şekil benzerliği hesaplanır.
        Eğer return_outliers True ise; (retVal, combined_score, similarity_dtw, similarity_procrustes, similarity_pearson, area_sim, shape_sim, prev_outliers, curr_outliers) döner.
        Aksi halde (retVal, combined_score, similarity_dtw, similarity_procrustes, similarity_pearson, area_sim, shape_sim) döner.
        """
        try:
            previous_series1 = previous_series.copy()
            current_series1 = current_series.copy()
            
            previous_series = self.normalize_shape(previous_series)
            current_series = self.normalize_shape(current_series)
            
            if self.check_for_nan_or_inf(previous_series) or self.check_for_nan_or_inf(current_series):
                return (2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [] , [])
            
            self.prev_outliers = self.identify_outliers(previous_series, distance_factor=3, small_gap_threshold=5)
            self.curr_outliers = self.identify_outliers(current_series, distance_factor=3, small_gap_threshold=5)
           
            cleaned_prev_s = self.remove_outliers_from_series(previous_series, self.prev_outliers)
            cleaned_curr_s = self.remove_outliers_from_series(current_series, self.curr_outliers)
            
            cleaned_prev_s = self.normalize_shape(cleaned_prev_s)
            cleaned_curr_s = self.normalize_shape(cleaned_curr_s)
            
            if len(cleaned_prev_s) < 8 or len(cleaned_curr_s) < 8:
                self.logger_series.warning("Aykırı değerlerin kaldırılmasından sonra yeterli nokta kalmadı.")
            
                return (2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [] , [])
            if self.check_for_nan_or_inf(cleaned_prev_s) or self.check_for_nan_or_inf(cleaned_curr_s):
                self.logger_series.warning("Yeniden örnekleme sonrası dizide NaN veya Inf değerler var.")
                return (2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [] , [])
            
            dtw_distance = self.calculate_dtw_distance(cleaned_prev_s, cleaned_curr_s)
            mtx1, mtx2, disparity = procrustes(cleaned_prev_s, cleaned_curr_s)
            pearson_corr = self.calculate_pearson_correlation(cleaned_prev_s, cleaned_curr_s)
            
            similarity_dtw = max(0, 100 - (dtw_distance / self.tolerance_dtw) * 100)
            similarity_procrustes = max(0, 100 - (disparity / self.tolerance_procrustes) * 100)
            similarity_pearson = pearson_corr
           
                  
            area_sim = self.area_similarity(cleaned_prev_s, cleaned_curr_s)
            #shape_sim = self.shape_similarity(cleaned_prev_s, cleaned_curr_s)
            iou_sim = self.iou_similarity(cleaned_prev_s, cleaned_curr_s, iou_power=3)
            shape_sim = iou_sim
            # Herhangi bir koşula girmezse oluşacak garanti skor hesabı
            #combined_score = (0.4 * similarity_dtw) + (0.3 * similarity_procrustes) + (0.3 * similarity_pearson)          
            
            
             #penalize new
            poly_prev = self.compute_convex_hull(cleaned_prev_s)
            poly_curr = self.compute_convex_hull(cleaned_curr_s)
            iou_value = self.compute_iou(poly_prev,poly_curr)
            similarity_penalized= self.compute_penalized_similarity(iou_value,  poly_prev,  poly_curr)
            # print("similarity_penalized Benzerliği :", similarity_penalized)
           
            similarity_pearson = similarity_penalized
            
            
                
            
        #    #test19 
        #    # # Belirli durumlar için farklı ağırlıklar uygulanıyor
        #     combined_score = (0.5 * similarity_dtw) + (0 * similarity_procrustes) + (0.5 * similarity_penalized)
        #     #Dtw 50 den küçük olma durumları
            
        #     #Dtw benzerliği 25 dan küçükse farklı olarak sınıflandırıyoruz
        #     if similarity_dtw <=25:
        #         combined_score = (0.8 * similarity_dtw) + (0.1 * similarity_procrustes) + (0.1 * similarity_penalized)    
        
        # #test23
        
        #    # # Belirli durumlar için farklı ağırlıklar uygulanıyor
        #     combined_score = (0.5 * similarity_dtw) + (0 * similarity_procrustes) + (0.5 * similarity_penalized)
        #     #Dtw 50 den küçük olma durumları
            
        #     #Dtw benzerliği 25 dan küçükse farklı olarak sınıflandırıyoruz
        #     if similarity_dtw <=25:
        #         combined_score = (0.8 * similarity_dtw) + (0.1 * similarity_procrustes) + (0.1 * similarity_penalized)    
              
           
        #     #Dtw 50 den büyük olma durumları
            
        #     if similarity_dtw >=70 : 
        #         if similarity_penalized < 20:
        #             combined_score = (0.4 * similarity_dtw) + (0.2 * similarity_procrustes) + (0.4 * similarity_penalized)  
        #         else:         
        #             combined_score = (0.6 * similarity_dtw) + (0.2 * similarity_procrustes) + (0.2 * similarity_penalized)       
           
            #Dtw 50 den büyük olma durumları
            
            # if similarity_dtw >=70 : #bu düşecek
            #       combined_score = (0.6 * similarity_dtw) + (0.2 * similarity_procrustes) + (0.2 * similarity_penalized)    
        
        #test 20    
            # combined_score = (0.5 * similarity_dtw) + (0 * similarity_procrustes) + (0.5 * similarity_penalized)
            #Dtw 50 den küçük olma durumları
            
            
            # #Dtw benzerliği 25 dan küçükse farklı olarak sınıflandırıyoruz
            # if similarity_dtw <=25:
            #     combined_score = (0.8 * similarity_dtw) + (0.1 * similarity_procrustes) + (0.1 * similarity_penalized)        
            
            # if similarity_dtw >=70 : 
            #       combined_score = (0.7 * similarity_dtw) + (0.2 * similarity_procrustes) + (0.1 * similarity_penalized)    
        #test24
        
        #    # # Belirli durumlar için farklı ağırlıklar uygulanıyor
        #     combined_score = (0.5 * similarity_dtw) + (0 * similarity_procrustes) + (0.5 * similarity_penalized)
        #     #Dtw 50 den küçük olma durumları
            
        #     #Dtw benzerliği 25 dan küçükse farklı olarak sınıflandırıyoruz
        #     if similarity_dtw <=25:
        #         combined_score = (0.8 * similarity_dtw) + (0.1 * similarity_procrustes) + (0.1 * similarity_penalized)    
              
              
           
        #     #Dtw 50 den büyük olma durumları
            
        #     if similarity_dtw >=70 : 
        #         if similarity_penalized < 20:
        #             combined_score = (0.4 * similarity_dtw) + (0.2 * similarity_procrustes) + (0.4 * similarity_penalized)  
        #         else:         
        #             combined_score = (0.6 * similarity_dtw) + (0.2 * similarity_procrustes) + (0.2 * similarity_penalized)
             
        #     if similarity_dtw >=80:
        #          combined_score = (0.7 * similarity_dtw) + (0.1 * similarity_procrustes) + (0.2 * similarity_penalized)      
        
        #test25
         # # Belirli durumlar için farklı ağırlıklar uygulanıyor
            combined_score = (0.5 * similarity_dtw) + (0 * similarity_procrustes) + (0.5 * similarity_penalized)
            #Dtw 50 den küçük olma durumları
            
            #Dtw benzerliği 25 dan küçükse farklı olarak sınıflandırıyoruz
            if similarity_dtw <=25:
                combined_score = (0.8 * similarity_dtw) + (0.1 * similarity_procrustes) + (0.1 * similarity_penalized)    
              
           
            #Dtw 50 den büyük olma durumları
            
            if similarity_dtw >=70 : 
                if similarity_penalized < 20:
                    combined_score = (0.4 * similarity_dtw) + (0.2 * similarity_procrustes) + (0.4 * similarity_penalized)  
                else:         
                    combined_score = (0.6 * similarity_dtw) + (0.2 * similarity_procrustes) + (0.2 * similarity_penalized)
                
                if similarity_penalized < 35 and similarity_procrustes < 10:
                    combined_score = (0.4 * similarity_dtw) + (0.2 * similarity_procrustes) + (0.4 * similarity_penalized)           
            if similarity_dtw >=80:
                 combined_score = (0.7 * similarity_dtw) + (0.1 * similarity_procrustes) + (0.2 * similarity_penalized)          
            
            retVal = 0 if combined_score >= 50 else 1
            if return_outliers:
                return (retVal, combined_score, similarity_dtw, similarity_procrustes, similarity_pearson, area_sim, shape_sim, self.prev_outliers, self.curr_outliers)
            else:
                return (retVal, combined_score, similarity_dtw, similarity_procrustes, similarity_pearson, area_sim, shape_sim)
        except Exception as e:
            self.logger_series.error(f"Seri Kontrol Hatası: {e}")
            if return_outliers:
                return (-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [] , [])
            else:
                return (-1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

###############################################################################
# 2) Logger Kurulumu
###############################################################################

def setup_logger():
    logger = logging.getLogger("CamCompareLogger")
    logger.setLevel(logging.INFO)
    logs_dir = os.path.join(os.getcwd(), "logs")
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, exist_ok=True)
    log_file = os.path.join(logs_dir, "CamCompare.log")
    handler = logging.FileHandler(log_file)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)
    return logger

###############################################################################
# 3) Görsel Oluşturarak Hata Kaydetme (plot_error)
###############################################################################

def plot_error(serA, serB, retVal, combined_score, similarity_dtw, similarity_procrustes, similarity_pearson,
               error_type, json_a_path, json_b_path, save_path, base_dir, outliers_a=None, outliers_b=None,
               area_sim=None, shape_sim=None):
    """
    İki seriyi çizerek bir görsel kaydeder.
    Başlıkta error_type, retVal, combined_score, similarity metriklerini ve
    alt kısımda Area & Shape Similarity bilgilerini (metin olarak) gösterir.
    Outlier noktaları yeşil renkle işaretlenir.
    """
    try:
        rel_a_path = os.path.relpath(json_a_path, start=base_dir)
        rel_b_path = os.path.relpath(json_b_path, start=base_dir)
    
        fig, ax = plt.subplots(figsize=(10,10))
        ax.plot(serA[:, 0], serA[:, 1], 'bo-', label='A')
        ax.plot(serB[:, 0], serB[:, 1], 'ro-', label='B')
    
        if outliers_a is not None and len(outliers_a) > 0:
            ax.plot(serA[outliers_a, 0], serA[outliers_a, 1], 'go', label='Outliers A')
        if outliers_b is not None and len(outliers_b) > 0:
            ax.plot(serB[outliers_b, 0], serB[outliers_b, 1], 'go', label='Outliers B')
        
        title_str = (f"{error_type}\n"
                     f"A: {rel_a_path} vs B: {rel_b_path}\n"
                     f"retVal={retVal}, combined_score={combined_score:.2f}\n"
                     f"similarity_dtw={similarity_dtw:.2f}, similarity_procrustes={similarity_procrustes:.2f}, similarity_pearson={similarity_pearson:.2f}")
        ax.set_title(title_str, fontsize=8)
        ax.legend()
        if area_sim is not None and shape_sim is not None:
            # Alt kısımda metin olarak ekleyelim (ax.transAxes kullanılarak)
            ax.text(0.5, -0.08, f"Area Similarity: {area_sim:.2f}   |   iou Similarity: {shape_sim:.2f}",
                    transform=ax.transAxes, ha="center", fontsize=8)
    
        plt.savefig(save_path)
        plt.close(fig)
        print(f"      => PNG kaydedildi: {save_path}")
    except Exception as e:
        print(f"      => Görsel kaydedilirken hata: {e}")

###############################################################################
# 4) Cam Karşılaştırma Fonksiyonları
###############################################################################

def compare_json_files_in_same_cam(json_list, matcher, out_cam_dir, logger, error_count_dict, folder_name, detailed_error_details, base_dir):
    if folder_name.endswith("-farklı"):
        msg = f"[INFO] Klasör adı '{folder_name}', kıyaslama yapılmayacak veya farklı şekilde işlenecek."
        print(f"      => {msg}")
        logger.info(msg)
        return

    n = len(json_list)
    error_dir = os.path.join(out_cam_dir, "errors")

    for i in range(n):
        for j in range(i + 1, n):
            json_a = json_list[i]
            json_b = json_list[j]

            if "current_series" not in json_a or "current_series" not in json_b:
                msg = f"[SAME_CAM_WARNING] current_series yok => skip. {json_a.get('json_path', '')} vs {json_b.get('json_path', '')}"
                print(f"      => {msg}")
                logger.warning(msg)
                continue

            curr_a = np.array(json_a["current_series"], dtype=float)
            curr_b = np.array(json_b["current_series"], dtype=float)

            # Artık yeni match_shapes 9 elemanlı tuple döndürecek (return_outliers=True)
            result = matcher.match_shapes(curr_a, curr_b, "same_cam", f"{i}_{j}", return_outliers=True)
            if len(result) == 9:
                (retVal, combined_score, similarity_dtw, similarity_procrustes, similarity_pearson,
                 area_sim, shape_sim, outliers_prev, outliers_curr) = result
            else:
                retVal, combined_score, similarity_dtw, similarity_procrustes, similarity_pearson, area_sim, shape_sim = result
                outliers_prev, outliers_curr = [], []

            if retVal != 0:
                error_count_dict["SAME_CAM_ERROR"] += 1
                error_type = "SAME_CAM_ERROR"
                msg = f"{error_type}: {json_a['json_path']} vs {json_b['json_path']}, retVal={retVal}, combined_score={combined_score:.2f}"
                print(f"      => Hata: {msg}")
                logger.error(msg)
                detailed_error_details['cam_errors'].append({
                    'error_type': error_type,
                    'cam_path': os.path.relpath(json_a['json_path'], start=base_dir),
                    'related_cam_path': os.path.relpath(json_b['json_path'], start=base_dir),
                    'retVal': retVal,
                    'combined_score': combined_score,
                    'similarity_dtw': similarity_dtw,
                    'similarity_procrustes': similarity_procrustes,
                    'similarity_pearson': similarity_pearson
                })
                nameA = os.path.splitext(os.path.basename(json_a["json_path"]))[0]
                nameB = os.path.splitext(os.path.basename(json_b["json_path"]))[0]
                out_name = f"{error_type}_{nameA}_vs_{nameB}_score{combined_score:.2f}_ret{retVal}.png"
                if not os.path.exists(error_dir):
                    os.makedirs(error_dir, exist_ok=True)
                save_path = os.path.join(error_dir, out_name)
                plot_error(
                    serA=matcher.normalize_shape(curr_a),
                    serB=matcher.normalize_shape(curr_b),
                    retVal=retVal,
                    combined_score=combined_score,
                    similarity_dtw=similarity_dtw,
                    similarity_procrustes=similarity_procrustes,
                    similarity_pearson=similarity_pearson,
                    error_type=error_type,
                    json_a_path=json_a["json_path"],
                    json_b_path=json_b["json_path"],
                    save_path=save_path,
                    base_dir=base_dir,
                    outliers_a=outliers_prev,
                    outliers_b=outliers_curr,
                    area_sim=area_sim,
                    shape_sim=shape_sim
                )

###############################################################################
# 5) Farklı Camlar Arası Kıyaslama Fonksiyonu
###############################################################################

def compare_with_other_cams(all_cam_json_lists, matcher, out_intermac_dir, intermac_logger, detailed_error_details, base_dir):
    error_count_dict = { "DIFF_CAM_ERROR": 0 }
    cam_names = sorted(all_cam_json_lists.keys())
    for i in range(len(cam_names)):
        for j in range(i+1, len(cam_names)):
            cam_i = cam_names[i]
            cam_j = cam_names[j]
            if cam_i.endswith('-farklı') and cam_j.endswith('-farklı'):
                continue
            cam_i_list = all_cam_json_lists[cam_i]
            cam_j_list = all_cam_json_lists[cam_j]
            out_cam_pair_dir = os.path.join(out_intermac_dir, f"{cam_i}_vs_{cam_j}", "errors")
            if not os.path.exists(out_cam_pair_dir):
                os.makedirs(out_cam_pair_dir, exist_ok=True)
            for json_a in cam_i_list:
                if "current_series" not in json_a:
                    msg = f"[DIFF_CAM_WARNING] current_series yok => skip. {json_a.get('json_path','')}"
                    print(f"      => {msg}")
                    intermac_logger.warning(msg)
                    continue
                curr_a = np.array(json_a["current_series"], dtype=float)
                for json_b in cam_j_list:
                    if "current_series" not in json_b:
                        msg = f"[DIFF_CAM_WARNING] current_series yok => skip. {json_b.get('json_path','')}"
                        print(f"      => {msg}")
                        intermac_logger.warning(msg)
                        continue
                    curr_b = np.array(json_b["current_series"], dtype=float)
                    result = matcher.match_shapes(curr_a, curr_b, "diff_cam", "diff", return_outliers=True)
                    if len(result) == 9:
                        (retVal, combined_score, similarity_dtw, similarity_procrustes, similarity_pearson,
                         area_sim, shape_sim, outliers_prev, outliers_curr) = result
                    else:
                        retVal, combined_score, similarity_dtw, similarity_procrustes, similarity_pearson, area_sim, shape_sim = result
                        outliers_prev, outliers_curr = [], []
                    if retVal == 0:
                        error_count_dict["DIFF_CAM_ERROR"] += 1
                        error_type = "DIFF_CAM_ERROR"
                        msg = f"{error_type}: {json_a['json_path']} vs {json_b['json_path']}, retVal=0, combined_score={combined_score:.2f}"
                        print(f"         => Hata: {msg}")
                        intermac_logger.error(msg)
                        detailed_error_details['diff_cam_errors'].append({
                            'error_type': error_type,
                            'cam_pair': f"{cam_i} vs {cam_j}",
                            'cam_a_path': os.path.relpath(json_a['json_path'], start=base_dir),
                            'cam_b_path': os.path.relpath(json_b['json_path'], start=base_dir),
                            'retVal': retVal,
                            'combined_score': combined_score,
                            'similarity_dtw': similarity_dtw,
                            'similarity_procrustes': similarity_procrustes,
                            'similarity_pearson': similarity_pearson
                        })
                        nameA = os.path.splitext(os.path.basename(json_a["json_path"]))[0]
                        nameB = os.path.splitext(os.path.basename(json_b["json_path"]))[0]
                        out_name = f"{error_type}_{nameA}_vs_{nameB}_score{combined_score:.2f}_ret0.png"
                        save_path = os.path.join(out_cam_pair_dir, out_name)
                        plot_error(
                            serA=matcher.normalize_shape(curr_a),
                            serB=matcher.normalize_shape(curr_b),
                            retVal=retVal,
                            combined_score=combined_score,
                            similarity_dtw=similarity_dtw,
                            similarity_procrustes=similarity_procrustes,
                            similarity_pearson=similarity_pearson,
                            error_type=error_type,
                            json_a_path=json_a["json_path"],
                            json_b_path=json_b["json_path"],
                            save_path=save_path,
                            base_dir=base_dir,
                            outliers_a=outliers_prev,
                            outliers_b=outliers_curr,
                            area_sim=area_sim,
                            shape_sim=shape_sim
                        )
    return error_count_dict

###############################################################################
# 6) Günlük Hata Raporunu Metin Formatında Oluşturma
###############################################################################

def generate_daily_error_report(detailed_error_details, day_output_path, day_name, day_total_comparisons, day_total_errors):
    report_path = os.path.join(day_output_path, f"{day_name}_daily_error_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Günlük Hata Özeti: {day_name}\n")
        f.write("="*50 + "\n\n")
        overall_error_rate = (day_total_errors / day_total_comparisons) * 100 if day_total_comparisons > 0 else 0
        f.write(f"Genel Hata Oranı: {overall_error_rate:.2f}%\n\n")
        f.write("Cam Bazında Hata Sayıları:\n")
        f.write("-"*50 + "\n")
        cam_error_counts = defaultdict(int)
        for error in detailed_error_details['cam_errors']:
            cam_error_counts[error['cam_path']] += 1
        for error in detailed_error_details['diff_cam_errors']:
            cam_error_counts[error['cam_a_path']] += 1
            cam_error_counts[error['cam_b_path']] += 1
        sorted_cams = sorted(cam_error_counts.items(), key=lambda item: item[1], reverse=True)
        for cam, count in sorted_cams:
            f.write(f"Cam: {cam}, Hata Sayısı: {count}\n")
        f.write("-"*50 + "\n")
    print(f"   => Daily Error Report saved: {report_path}")

###############################################################################
# 7) Gün Bazında Özet Fonksiyonu (Bar Chart)
###############################################################################

def plot_day_summary(day_error_summary, day_name, out_day_summary_png):
    if not day_error_summary:
        return
    labels = []
    same_values = []
    diff_values = []
    for intermac, ecounts in day_error_summary.items():
        labels.append(intermac)
        same_values.append(ecounts.get("SAME_CAM_ERROR", 0))
        diff_values.append(ecounts.get("DIFF_CAM_ERROR", 0))
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10,5))
    rects1 = ax.bar(x - width/2, same_values, width, label='SAME_CAM_ERROR')
    rects2 = ax.bar(x + width/2, diff_values, width, label='DIFF_CAM_ERROR')
    ax.set_ylabel('Error Count')
    ax.set_title(f'{day_name} - Intermac Summary of Errors')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_day_summary_png)
    plt.close(fig)
    print(f"   => Day Summary saved: {out_day_summary_png}")

###############################################################################
# 8) IntermacX Bazında Hata Özetleri (Detaylı Rapor)
###############################################################################

def collect_intermac_errors(all_cam_json_lists, matcher, intermac_output_path, logger, detailed_error_details, base_dir):
    cam_error_details = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    cam_names = sorted(all_cam_json_lists.keys())
    for i in range(len(cam_names)):
        for j in range(i+1, len(cam_names)):
            cam_i = cam_names[i]
            cam_j = cam_names[j]
            if cam_i.endswith('-farklı') and cam_j.endswith('-farklı'):
                continue
            cam_i_list = all_cam_json_lists[cam_i]
            cam_j_list = all_cam_json_lists[cam_j]
            for json_a in cam_i_list:
                if "current_series" not in json_a:
                    logger.warning(f"[COLLECT_WARNING] current_series yok => skip. {json_a.get('json_path','')}")
                    continue
                curr_a = np.array(json_a["current_series"], dtype=float)
                for json_b in cam_j_list:
                    if "current_series" not in json_b:
                        logger.warning(f"[COLLECT_WARNING] current_series yok => skip. {json_b.get('json_path','')}")
                        continue
                    curr_b = np.array(json_b["current_series"], dtype=float)
                    result = matcher.match_shapes(curr_a, curr_b, "collect_errors", "collect", return_outliers=True)
                    if len(result) == 9:
                        (retVal, combined_score, similarity_dtw, similarity_procrustes, similarity_pearson,
                         area_sim, shape_sim, _, _) = result
                    else:
                        retVal, combined_score, similarity_dtw, similarity_procrustes, similarity_pearson, area_sim, shape_sim = result
                    if retVal != 0:
                        if retVal == 1:
                            error_type = "retVal=1 (Farklı)"
                        elif retVal == 2:
                            error_type = "retVal=2 (Veri Hatalı)"
                        elif retVal == -1:
                            error_type = "retVal=-1 (Exception)"
                        else:
                            error_type = f"retVal={retVal}"
                        cam_error_details[cam_i][cam_j][error_type] += 1
                        cam_error_details[cam_j][cam_i][error_type] += 1
    ERROR_THRESHOLD = 3
    significant_errors = {}
    for cam, related_cams in cam_error_details.items():
        total_errors = sum([sum(error_types.values()) for error_types in related_cams.values()])
        if total_errors > ERROR_THRESHOLD:
            significant_errors[cam] = {
                "total_errors": total_errors,
                "related_cams": related_cams
            }
    if not significant_errors:
        print("   => Hiçbir cam belirlenen hata eşiğinin üzerinde hata almamış.")
        return
    sorted_significant_errors = sorted(significant_errors.items(), key=lambda item: item[1]["total_errors"], reverse=True)
    report_path = os.path.join(intermac_output_path, "significant_cam_errors_detail.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"IntermacX - Yüksek Hata Oranına Sahip Camlar (Eşik: {ERROR_THRESHOLD})\n")
        f.write("="*80 + "\n\n")
        for cam, details in sorted_significant_errors:
            f.write(f"Cam: {cam}\n")
            f.write(f"Toplam Hata Sayısı: {details['total_errors']}\n")
            f.write("Hatalar:\n")
            for related_cam, error_types in details['related_cams'].items():
                for error_type, count in error_types.items():
                    f.write(f"  - {related_cam}: {error_type} - {count} hata\n")
            f.write("-"*80 + "\n\n")
    print(f"   => Significant Cam Errors Detail Report saved: {report_path}")

###############################################################################
# 9) IntermacX Bazında Hata Özet Fonksiyonu (Bar Chart)
###############################################################################

def plot_intermac_summary(intermac_error_summary, summary_plot_path):
    if not intermac_error_summary:
        return
    labels = []
    same_values = []
    diff_values = []
    for intermac, ecounts in intermac_error_summary.items():
        labels.append(intermac)
        same_values.append(ecounts.get("SAME_CAM_ERROR", 0))
        diff_values.append(ecounts.get("DIFF_CAM_ERROR", 0))
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10,5))
    rects1 = ax.bar(x - width/2, same_values, width, label='SAME_CAM_ERROR')
    rects2 = ax.bar(x + width/2, diff_values, width, label='DIFF_CAM_ERROR')
    ax.set_ylabel('Error Count')
    ax.set_title('Intermac Summary of Errors')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(summary_plot_path)
    plt.close(fig)
    print(f"   => Intermac Summary Plot saved: {summary_plot_path}")

###############################################################################
# 10) Genel Performans Değerlendirmesi
###############################################################################

def evaluate_performance(total_comparisons, total_errors, out_intermac_dir, day_error_rates):
    performance_report_path = os.path.join(out_intermac_dir, "performance_evaluation.txt")
    with open(performance_report_path, 'w', encoding='utf-8') as f:
        f.write("Genel Performans Değerlendirmesi\n")
        f.write("="*50 + "\n\n")
        overall_error_rate = (total_errors / total_comparisons) * 100 if total_comparisons > 0 else 0
        f.write(f"Toplam Karşılaştırma Sayısı: {int(total_comparisons)}\n")
        f.write(f"Toplam Hata Sayısı: {int(total_errors)}\n")
        f.write(f"Genel Hata Oranı: {overall_error_rate:.2f}%\n\n")
        f.write("Gün Bazında Hata Oranları:\n")
        f.write("-"*50 + "\n")
        for day, rate in day_error_rates.items():
            f.write(f"Gün: {day}, Hata Oranı: {rate:.2f}%\n")
        f.write("-"*50 + "\n")
    print(f"   => Performance Evaluation Report saved: {performance_report_path}")

###############################################################################
# 11) Gün Bazında Özet Fonksiyonu (Bar Chart)
###############################################################################

def plot_day_summary(day_error_summary, day_name, out_day_summary_png):
    if not day_error_summary:
        return
    labels = []
    same_values = []
    diff_values = []
    for intermac, ecounts in day_error_summary.items():
        labels.append(intermac)
        same_values.append(ecounts.get("SAME_CAM_ERROR", 0))
        diff_values.append(ecounts.get("DIFF_CAM_ERROR", 0))
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10,5))
    rects1 = ax.bar(x - width/2, same_values, width, label='SAME_CAM_ERROR')
    rects2 = ax.bar(x + width/2, diff_values, width, label='DIFF_CAM_ERROR')
    ax.set_ylabel('Error Count')
    ax.set_title(f'{day_name} - Intermac Summary of Errors')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_day_summary_png)
    plt.close(fig)
    print(f"   => Day Summary saved: {out_day_summary_png}")

###############################################################################
# 12) Main Fonksiyonu
###############################################################################

def main():
    logger = setup_logger()
    logx_input = "E:/Proje repo genel/3001-works/series_check/temiz_cam_dataset/small"
    logx_output = "E:/Proje repo genel/3001-works/series_check/All_tests/20ocak/Test25-3001-org-cleardataset-penalized-sistem"
    if not os.path.exists(logx_input):
        logger.warning(f"Girdi yok: {logx_input}")
        print(f"Girdi yok: {logx_input}")
        return
    if not os.path.exists(logx_output):
        os.makedirs(logx_output, exist_ok=True)
        logger.info(f"Output dizini oluşturuldu: {logx_output}")
    print(f"Girdi: {logx_input}")
    print(f"Çıktı: {logx_output}")

    shape_matcher = ShapeMatcher(
        tolerance_dtw=3000,
        tolerance_procrustes=0.06,
        tolerance_pearson=100,
        num_points=100,
    )

    total_comparisons = 0
    total_errors = 0
    day_error_rates = {}

    day_folders = [d for d in os.listdir(logx_input) if os.path.isdir(os.path.join(logx_input, d))]
    for day_name in sorted(day_folders):
        day_input_path = os.path.join(logx_input, day_name)
        day_output_path = os.path.join(logx_output, day_name)
        if not os.path.exists(day_output_path):
            os.makedirs(day_output_path, exist_ok=True)
        intermac_folders = [f for f in os.listdir(day_input_path) if os.path.isdir(os.path.join(day_input_path, f))]
        day_error_summary = {}
        detailed_error_details = {
            'cam_errors': [],
            'diff_cam_errors': []
        }
        day_total_comparisons = 0
        day_total_errors = 0
        for intermac_name in sorted(intermac_folders):
            intermac_input_path = os.path.join(day_input_path, intermac_name)
            intermac_output_path = os.path.join(day_output_path, intermac_name)
            if not os.path.exists(intermac_output_path):
                os.makedirs(intermac_output_path, exist_ok=True)
            print(f"\n=== Gün: {day_name}, Intermac: {intermac_name} ===")
            logger.info(f"Gün: {day_name}, Intermac: {intermac_name}")
            cam_folders = [c for c in os.listdir(intermac_input_path) if os.path.isdir(os.path.join(intermac_input_path, c))]
            all_cam_json_lists = {}
            intermac_error_summary = {}
            for cam_name in sorted(cam_folders):
                cam_input_path = os.path.join(intermac_input_path, cam_name)
                cam_output_path = os.path.join(intermac_output_path, cam_name)
                if not os.path.exists(cam_output_path):
                    os.makedirs(cam_output_path, exist_ok=True)
                print(f"--- Cam: {cam_name} ---")
                logger.info(f"Cam klasörü: {cam_input_path}")
                json_files = [f for f in os.listdir(cam_input_path) if f.endswith(".json")]
                cam_json_list = []
                for jf in json_files:
                    fullp = os.path.join(cam_input_path, jf)
                    try:
                        with open(fullp, 'r', encoding='utf-8') as ff:
                            dat = json.load(ff)
                            dat["json_path"] = fullp
                            dat["imageName"] = dat.get("imageName", jf)
                            cam_json_list.append(dat)
                    except Exception as e:
                        msg = f"JSON yükleme hatası: {fullp} => {e}"
                        print(f"   => {msg}")
                        logger.error(msg)
                error_count_dict = { "SAME_CAM_ERROR": 0 }
                compare_json_files_in_same_cam(
                    json_list=cam_json_list,
                    matcher=shape_matcher,
                    out_cam_dir=cam_output_path,
                    logger=logger,
                    error_count_dict=error_count_dict,
                    folder_name=cam_name,
                    detailed_error_details=detailed_error_details,
                    base_dir=logx_input
                )
                intermac_error_summary[cam_name] = { "SAME_CAM_ERROR": error_count_dict["SAME_CAM_ERROR"] }
                comparisons = (len(cam_json_list) * (len(cam_json_list) - 1)) / 2
                day_total_comparisons += comparisons
                day_total_errors += error_count_dict["SAME_CAM_ERROR"]
                all_cam_json_lists[cam_name] = cam_json_list
            diff_err_dict = compare_with_other_cams(
                all_cam_json_lists=all_cam_json_lists,
                matcher=shape_matcher,
                out_intermac_dir=intermac_output_path,
                intermac_logger=logger,
                detailed_error_details=detailed_error_details,
                base_dir=logx_input
            )
            intermac_error_summary["cam_pairs"] = { "DIFF_CAM_ERROR": diff_err_dict["DIFF_CAM_ERROR"] }
            day_total_errors += diff_err_dict["DIFF_CAM_ERROR"]
            cam_names = list(all_cam_json_lists.keys())
            for i_cam in cam_names:
                for j_cam in cam_names:
                    if i_cam < j_cam and not (i_cam.endswith('-farklı') and j_cam.endswith('-farklı')):
                        comparisons = len(all_cam_json_lists[i_cam]) * len(all_cam_json_lists[j_cam])
                        day_total_comparisons += comparisons
            plot_intermac_summary(intermac_error_summary, os.path.join(intermac_output_path, f"{intermac_name}_summary.png"))
            collect_intermac_errors(
                all_cam_json_lists=all_cam_json_lists,
                matcher=shape_matcher,
                intermac_output_path=intermac_output_path,
                logger=logger,
                detailed_error_details=detailed_error_details,
                base_dir=logx_input
            )
            day_error_summary[intermac_name] = intermac_error_summary
            generate_daily_error_report(
                detailed_error_details=detailed_error_details,
                day_output_path=day_output_path,
                day_name=day_name,
                day_total_comparisons=day_total_comparisons,
                day_total_errors=day_total_errors
            )
            day_error_rate = (day_total_errors / day_total_comparisons) * 100 if day_total_comparisons > 0 else 0
            day_error_rates[day_name] = day_error_rate
            day_summary_plot_path = os.path.join(day_output_path, f"{day_name}_day_summary.png")
            plot_day_summary(
                day_error_summary=day_error_summary,
                day_name=day_name,
                out_day_summary_png=day_summary_plot_path
            )
            total_comparisons += day_total_comparisons
            total_errors += day_total_errors

    evaluate_performance(
        total_comparisons=total_comparisons,
        total_errors=total_errors,
        out_intermac_dir=logx_output,
        day_error_rates=day_error_rates
    )
    print("\nTüm işlemler tamamlandı.")

###############################################################################
# 13) Çalıştırma
###############################################################################

if __name__ == "__main__":
    main()

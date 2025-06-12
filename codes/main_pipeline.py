import cv2
import numpy as np
from collections import deque
import torch
import torch.nn as nn
from ultralytics import YOLO
import time
import os
import csv
from datetime import datetime

class VideoLogger:
    def __init__(self):
        self.log_data = []
        self.start_time = time.time()
        self.current_timestamp = 0
        
        # Output klasörünü oluştur
        os.makedirs("output_videos", exist_ok=True)
        
    def log_event(self, event_type, details):
        """Event log ekle"""
        current_time = time.time()
        timestamp = current_time - self.start_time
        
        log_entry = {
            'timestamp': round(timestamp, 2),
            'event_type': event_type,
            'details': details
        }
        self.log_data.append(log_entry)
        
    def save_log(self):
        """Log'u CSV olarak kaydet"""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output_videos/runway_test_log_{timestamp_str}.csv"
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'event_type', 'details']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for entry in self.log_data:
                writer.writerow(entry)
        
        print(f"Log dosyası kaydedildi: {filename}")
        return filename

class KalmanMaskTracker:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                 [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
        self.initialized = False

    def update_tracking(self, mask_center):
        """Mask merkez noktasını track et"""
        if not self.initialized and mask_center is not None:
            # İlk kez başlatma
            self.kalman.statePre = np.array([mask_center[0], mask_center[1], 0, 0], dtype=np.float32)
            self.kalman.statePost = np.array([mask_center[0], mask_center[1], 0, 0], dtype=np.float32)
            self.initialized = True
            return mask_center

        # Prediction
        prediction = self.kalman.predict()

        # Update with measurement
        if mask_center is not None:
            measurement = np.array([[mask_center[0]], [mask_center[1]]], dtype=np.float32)
            self.kalman.correct(measurement)
            return mask_center
        else:
            # Prediction kullan
            return (int(prediction[0]), int(prediction[1]))

    def get_mask_center(self, mask):
        """Mask'ın merkez noktasını bul"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return (cx, cy)
        return None


class TemporalMaskSmoother:
    def __init__(self, buffer_size=5, blend_alpha=0.7):
        """
        buffer_size: Kaç frame'i hafızada tutacak
        blend_alpha: Geçmiş mask'lerle karıştırma oranı
        """
        self.buffer_size = buffer_size
        self.blend_alpha = blend_alpha
        self.mask_buffer = deque(maxlen=buffer_size)

    def add_mask(self, mask):
        """Yeni mask ekle"""
        self.mask_buffer.append(mask.copy())

    def temporal_smooth(self, current_mask):
        """Zamansal pürüzleştirme uygula"""
        if len(self.mask_buffer) < 2:
            return current_mask

        # Mevcut ve önceki mask'leri al
        prev_mask = self.mask_buffer[-2]

        # Weighted blending
        smoothed_mask = (
            self.blend_alpha * current_mask.astype(np.float32) +
            (1 - self.blend_alpha) * prev_mask.astype(np.float32)
        )

        # Threshold uygula
        smoothed_mask = (smoothed_mask > 127).astype(np.uint8) * 255

        return smoothed_mask

    def process_video_frame(self, raw_mask):
        """Video frame işleme pipeline'ı"""
        # Mask'ı buffer'a ekle
        self.add_mask(raw_mask)

        # Zamansal pürüzleştirme uygula
        smoothed_mask = self.temporal_smooth(raw_mask)

        return smoothed_mask

    def get_smoothed_mask(self):
         """Buffer'daki mask'leri kullanarak pürüzleştirilmiş mask döndür"""
         if not self.mask_buffer:
             return None

         # buffer'daki tüm mask'lerin ortalamasını al
         avg_mask = np.mean(list(self.mask_buffer), axis=0)

         # Threshold uygula
         smoothed_mask = (avg_mask > 127).astype(np.uint8) * 255

         return smoothed_mask

class RunwayDetectionPipeline:
    def __init__(self, yolo_model_path, unet_model_path):
        """
        yolo_model_path: YOLO pist tespit modeli yolu
        unet_model_path: U-Net pist segmentasyon modeli yolu
        """
        # Modelleri yükle
        self.yolo_model = YOLO(yolo_model_path)
        
        # U-Net modeli - CPU'da yükle
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # U-Net modelini yükle (CPU/GPU otomatik seçim)
        try:
            # Önce tam model olarak yüklemeyi dene
            self.unet_model = torch.load(unet_model_path, map_location=self.device)
            
            # Eğer OrderedDict ise (sadece state_dict), model mimarisini oluştur
            if isinstance(self.unet_model, dict):
                print("Model state_dict olarak kaydedilmiş, model mimarisi oluşturuluyor...")
                self.unet_model = self.create_unet_model()  # Model mimarisini oluştur
                self.unet_model.load_state_dict(torch.load(unet_model_path, map_location=self.device))
            
            self.unet_model.eval()
            self.unet_model.to(self.device)
            
            # Model optimizasyonu
            if self.device.type == 'cuda':
                self.unet_model = torch.jit.script(self.unet_model)  # JIT compile
            
        except Exception as e:
            print(f"U-Net model yüklenirken hata: {e}")
            print("Demo için sahte segmentasyon mask'i kullanılıyor...")
            self.unet_model = None
        
        # Pipeline bileşenleri
        self.kalman_tracker = KalmanMaskTracker()
        self.mask_smoother = TemporalMaskSmoother(buffer_size=3, blend_alpha=0.8)
        
        # Logger ekle
        self.logger = VideoLogger()
        
        # Performans optimizasyonu - Segmentasyon için
        self.segmentation_skip_frames = 2  # Her 2 frame'de bir segmentasyon çalıştır
        self.segmentation_counter = 0
        self.last_segmentation_mask = None
        
        # Performans optimizasyonu
        self.frame_skip_counter = 0
        self.yolo_skip_frames = 3  # Her 3 frame'de bir YOLO çalıştır
        self.last_yolo_result = None
        self.input_resolution = (640, 640)  # YOLO için küçük resolution
        
        # Durum değişkenleri
        self.runway_detected = False
        self.in_segmentation_mode = False
        self.last_bbox = None
        self.detection_confidence_threshold = 0.5
        self.frames_without_detection = 0
        self.max_frames_without_detection = 10
    
    def create_unet_model(self):
        class UNet(nn.Module):
            def __init__(self, in_channels=3, out_channels=1):
                super(UNet, self).__init__()

                # Encoder
                self.enc1 = self.conv_block(in_channels, 64)
                self.enc2 = self.conv_block(64, 128)
                self.enc3 = self.conv_block(128, 256)
                self.enc4 = self.conv_block(256, 512)

                # Bottleneck
                self.bottleneck = self.conv_block(512, 1024)

                # Decoder
                self.dec4 = self.conv_block(1024 + 512, 512)
                self.dec3 = self.conv_block(512 + 256, 256)
                self.dec2 = self.conv_block(256 + 128, 128)
                self.dec1 = self.conv_block(128 + 64, 64)

                self.final = nn.Conv2d(64, out_channels, 1)
                self.sigmoid = nn.Sigmoid()

            def conv_block(self, in_ch, out_ch):
                return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU(inplace=True)
            )

            def forward(self, x):
                # Encoder path
                e1 = self.enc1(x)
                e2 = self.enc2(nn.MaxPool2d(2)(e1))
                e3 = self.enc3(nn.MaxPool2d(2)(e2))
                e4 = self.enc4(nn.MaxPool2d(2)(e3))

                # Bottleneck
                b = self.bottleneck(nn.MaxPool2d(2)(e4))

                # Decoder path with skip connections
                d4 = self.dec4(torch.cat([nn.Upsample(scale_factor=2)(b), e4], 1))
                d3 = self.dec3(torch.cat([nn.Upsample(scale_factor=2)(d4), e3], 1))
                d2 = self.dec2(torch.cat([nn.Upsample(scale_factor=2)(d3), e2], 1))
                d1 = self.dec1(torch.cat([nn.Upsample(scale_factor=2)(d2), e1], 1))

                return self.final(d1)
        return UNet()
    
    def create_demo_mask(self, frame):
        """Demo için sahte segmentasyon mask'i oluştur"""
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Frame merkezinden başlayarak pist benzeri bir mask oluştur
        center_x = width // 2
        center_y = height // 2
        
        # Trapez şeklinde pist mask'i
        pts = np.array([
            [center_x - 100, height],
            [center_x + 100, height],
            [center_x + 50, center_y],
            [center_x - 50, center_y]
        ], np.int32)
        
        cv2.fillPoly(mask, [pts], 255)
        
        return mask

    def detect_runway_yolo(self, frame):
        """YOLO ile pist tespiti - Optimized"""
        # Frame skip logic - performans için
        if self.frame_skip_counter < self.yolo_skip_frames and self.last_yolo_result is not None:
            self.frame_skip_counter += 1
            return self.last_yolo_result
        
        self.frame_skip_counter = 0
        
        # Frame'i küçült - YOLO için
        small_frame = cv2.resize(frame, self.input_resolution)
        
        # YOLO inference
        results = self.yolo_model(small_frame, verbose=False)  # verbose=False for speed
        
        # Scale faktörü hesapla
        scale_x = frame.shape[1] / self.input_resolution[0]
        scale_y = frame.shape[0] / self.input_resolution[1]
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    confidence = box.conf[0].item()
                    if confidence > self.detection_confidence_threshold:
                        # Bbox koordinatları - orijinal boyuta scale et
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)
                        
                        result_tuple = (True, (x1, y1, x2, y2), confidence)
                        self.last_yolo_result = result_tuple
                        return result_tuple
        
        result_tuple = (False, None, 0.0)
        self.last_yolo_result = result_tuple
        return result_tuple
    
    def segment_runway_unet(self, frame):
        """U-Net ile pist segmentasyonu - Performans optimizasyonu ile"""
        # Frame skip kontrolü - segmentasyon ağır işlem
        if self.segmentation_counter < self.segmentation_skip_frames and self.last_segmentation_mask is not None:
            self.segmentation_counter += 1
            return self.last_segmentation_mask
        
        self.segmentation_counter = 0
        
        # Eğer model yüklenmediyse demo mask döndür
        if self.unet_model is None:
            demo_mask = self.create_demo_mask(frame)
            self.last_segmentation_mask = demo_mask
            return demo_mask
        
        # Orijinal frame boyutları
        original_height, original_width = frame.shape[:2]
        
        # Model için preprocessing - daha küçük boyut performans için
        input_size = (128, 128)  # 256'dan 128'e düşürüldü
        resized_frame = cv2.resize(frame, input_size)
        
        # Normalizasyon (RGB'ye çevir)
        if len(resized_frame.shape) == 3:
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
        normalized_frame = resized_frame.astype(np.float32) / 255.0
        
        # Model inference
        try:
            with torch.no_grad():
                # Tensor'a çevir ve device'a gönder
                input_tensor = torch.from_numpy(normalized_frame).permute(2, 0, 1).unsqueeze(0).to(self.device)
                
                # Model çıktısı al
                output = self.unet_model(input_tensor)
                
                # Sigmoid ve CPU'ya geri getir
                mask = torch.sigmoid(output).squeeze().cpu().numpy()
                
                # Binary mask'a çevir
                mask = (mask > 0.5).astype(np.uint8) * 255
                
        except Exception as e:
            print(f"U-Net inference hatası: {e}")
            demo_mask = self.create_demo_mask(frame)
            self.last_segmentation_mask = demo_mask
            return demo_mask
        
        # Mask boyutunu kontrol et
        if len(mask.shape) == 2:
            mask_height, mask_width = mask.shape
        else:
            print(f"Beklenmeyen mask boyutu: {mask.shape}")
            demo_mask = self.create_demo_mask(frame)
            self.last_segmentation_mask = demo_mask
            return demo_mask
        
        # Orijinal frame boyutuna resize et
        mask_resized = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_NEAREST)
        
        # Cache'le
        self.last_segmentation_mask = mask_resized
        
        return mask_resized
    
    def check_runway_start(self, frame, bbox):
        """Pistin başına gelinip gelinmediğini kontrol et"""
        # Basit heuristic: bbox'ın frame'in alt kısmına yaklaşması
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]
        
        x1, y1, x2, y2 = bbox
        bbox_bottom = y2
        bbox_width = x2 - x1
        
        # Pist frame'in alt yarısında ve yeterince geniş ise segmentasyon moduna geç
        if (bbox_bottom > frame_height * 0.6 and 
            bbox_width > frame_width * 0.4):
            return True
        
        return False
    
    def process_frame(self, frame):
        """Ana frame işleme fonksiyonu"""
        output_frame = frame.copy()
        status_text = ""
        
        if not self.in_segmentation_mode:
            # YOLO tespit modunda
            runway_found, bbox, confidence = self.detect_runway_yolo(frame)
            
            if runway_found:
                self.runway_detected = True
                self.last_bbox = bbox
                self.frames_without_detection = 0
                
                # Bbox çiz
                x1, y1, x2, y2 = bbox
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Confidence göster
                cv2.putText(output_frame, f'Runway: {confidence:.2f}', 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                status_text = "Pist Tespit Edildi"
                
                # Pistin başına gelinip gelinmediğini kontrol et
                if self.check_runway_start(frame, bbox):
                    self.in_segmentation_mode = True
                    status_text = "Segmentasyon Moduna Geçiliyor..."
                    
            else:
                self.frames_without_detection += 1
                if self.frames_without_detection > self.max_frames_without_detection:
                    self.runway_detected = False
                
                if self.runway_detected:
                    status_text = "Pist Tespit Edildi (Tracking)"
                else:
                    status_text = "Pist Aranıyor..."
        
        else:
            # Segmentasyon modunda
            raw_mask = self.segment_runway_unet(frame)
            
            # Mask boyutu kontrolü
            if raw_mask.shape[:2] != frame.shape[:2]:
                print(f"UYARI: Mask boyutu uyumsuz! Frame: {frame.shape[:2]}, Mask: {raw_mask.shape[:2]}")
                raw_mask = cv2.resize(raw_mask, (frame.shape[1], frame.shape[0]))
            
            # Temporal smoothing uygula
            smoothed_mask = self.mask_smoother.process_video_frame(raw_mask)
            
            # Kalman tracking
            mask_center = self.kalman_tracker.get_mask_center(smoothed_mask)
            tracked_center = self.kalman_tracker.update_tracking(mask_center)
            
            # Mask'ı frame'e uygula - düzeltilmiş
            if len(smoothed_mask.shape) == 2:
                # 2D mask'ı 3 kanala çevir
                mask_colored = cv2.applyColorMap(smoothed_mask, cv2.COLORMAP_JET)
                
                # Boyut uyumunu kontrol et
                if mask_colored.shape[:2] == output_frame.shape[:2]:
                    output_frame = cv2.addWeighted(output_frame, 0.7, mask_colored, 0.3, 0)
                else:
                    print(f"Mask uygulama hatası - Frame: {output_frame.shape}, Mask: {mask_colored.shape}")
            
            # Tracked center'ı göster
            if tracked_center:
                cv2.circle(output_frame, tracked_center, 5, (255, 255, 255), -1)
                cv2.circle(output_frame, tracked_center, 15, (255, 255, 255), 2)
                
                # Center koordinatlarını göster
                cv2.putText(output_frame, f'Center: ({tracked_center[0]}, {tracked_center[1]})', 
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            status_text = "Pist Segmentasyonu Aktif"
        
        # Status text'i ekle
        cv2.putText(output_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        return output_frame
    
    def reset_pipeline(self):
        """Pipeline'ı sıfırla"""
        self.runway_detected = False
        self.in_segmentation_mode = False
        self.last_bbox = None
        self.frames_without_detection = 0
        self.kalman_tracker = KalmanMaskTracker()
        self.mask_smoother = TemporalMaskSmoother(buffer_size=5, blend_alpha=0.7)
        
        # Segmentasyon cache'ini temizle
        self.last_segmentation_mask = None
        self.segmentation_counter = 0
        self._segmentation_logged = False
        
        # Log event
        self.logger.log_event("PIPELINE_RESET", "Pipeline manually reset")

def main():
    """Ana çalıştırma fonksiyonu"""

    yolo_model_path = r"trained_models\best.pt"
    unet_model_path = r"trained_models\unet_final.pth"
    
     # Pipeline başlat
    pipeline = RunwayDetectionPipeline(yolo_model_path, unet_model_path)
    
    # Logger'ı başlat
    pipeline.logger.log_event("PIPELINE_START", "Video processing started")
    
    cap = cv2.VideoCapture(r"videos\input_videos\landing_video_6_goog.mp4") 
    
    # Video ayarları - performans için
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print("Pipeline başlatıldı. Tuş komutları:")
    print("- 'q': Çıkış")
    print("- 'r': Pipeline sıfırla")
    print("- 's': Segmentasyon modunu zorla")
    print("- 'd': Detection moduna geri dön")
    
    # FPS sayacı
    fps_counter = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame okunamadı!")
            break
        
        # FPS hesaplama
        fps_counter += 1
        if fps_counter % 30 == 0:
            elapsed_time = time.time() - start_time
            current_fps = 30 / elapsed_time
            print(f"Current FPS: {current_fps:.1f}")
            start_time = time.time()
        
        # Frame'i işle
        processed_frame = pipeline.process_frame(frame)
        
        # Sonucu göster
        cv2.imshow('Runway Detection & Tracking Pipeline', processed_frame)
        
        # Çıkış kontrolü
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            pipeline.logger.log_event("USER_INPUT", "Exit requested (q)")
            break
        elif key == ord('r'):
            # Pipeline'ı sıfırla
            pipeline.reset_pipeline()
            print("Pipeline sıfırlandı!")
        elif key == ord('s'):
            # Segmentasyon modunu zorla
            pipeline.in_segmentation_mode = True
            pipeline.logger.log_event("USER_INPUT", "Segmentation mode forced (s)")
            print("Segmentasyon modu zorlandı!")
        elif key == ord('d'):
            # Detection moduna geri dön
            pipeline.in_segmentation_mode = False
            pipeline.logger.log_event("USER_INPUT", "Detection mode activated (d)")
            print("Detection moduna dönüldü!")
        elif key == ord('p'):
            # Performans modu toggle
            pipeline.yolo_skip_frames = 5 if pipeline.yolo_skip_frames == 3 else 3
            pipeline.logger.log_event("USER_INPUT", f"Performance mode toggle (p) - skip frames: {pipeline.yolo_skip_frames}")
            print(f"YOLO skip frames: {pipeline.yolo_skip_frames}")
    
    # Log'u kaydet ve temizlik
    pipeline.logger.log_event("PIPELINE_END", "Video processing completed")
    log_file = pipeline.logger.save_log()
    print(f"Test tamamlandı. Log dosyası: {log_file}")
    
    # Temizlik
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

import cv2
import numpy as np
import streamlit as st
import os
from datetime import datetime
import queue
import time
import base64
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from main_pipeline import RunwayDetectionPipeline, VideoLogger
from session_utils import init_session_state

init_session_state()

# Gerçek zamanlı video işleme 
# RunwayDetectionPipeline ile ortak çalışır
class StreamlitVideoProcessor:
    def __init__(self):
        self.processing = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.status_queue = queue.Queue()
        self.current_frame = None
        self.stop_processing = False
        self.metrics = {
            'runway_detected': False,
            'segmentation_active': False,
            'deviation': 0,
            'confidence': 0.0,
            'mode': 'Searching...'
        }

    # Tek bir kareyi işler ve metrikleri günceller
    def process_frame_realtime(self, frame, pipeline):
        """Tek frame işleme - gerçek zamanlı için optimize edilmiş"""
        try:
            processed_frame = pipeline.process_frame(frame)
            
            # Metrics güncelle
            self.update_metrics_from_pipeline(pipeline)
            
            return processed_frame
            
        except Exception as e:
            st.error(f"Frame işleme hatası: {str(e)}")
            return frame
    
    # Canlı video akışı üzerinden kareleri işleyerek frame-by-frame çıktı verir
    def process_video_stream(self, video_path, yolo_model_path, unet_model_path, 
                           frame_callback=None, metrics_callback=None):
        """
        Gerçek zamanlı video stream işleme
        frame_callback: Her frame için çağrılacak fonksiyon
        metrics_callback: Metrics güncellemesi için çağrılacak fonksiyon
        """
        
        # Pipeline başlat
        pipeline = RunwayDetectionPipeline(yolo_model_path, unet_model_path)
        pipeline.logger.log_event("REALTIME_START", f"Real-time processing started: {video_path}")
        
        # Video input
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise Exception(f"Video açılamadı: {video_path}")
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_count = 0
        self.processing = True
        self.stop_processing = False
        
        try:
            while self.processing and not self.stop_processing:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Frame işle
                processed_frame = self.process_frame_realtime(frame, pipeline)
                
                # Callback'leri çağır
                if frame_callback:
                    frame_callback(processed_frame, frame_count, total_frames)
                
                if metrics_callback:
                    metrics_callback(self.metrics)
                
                frame_count += 1
                
                # FPS kontrolü (gerçek zamanlı hissi için)
                time.sleep(1.0 / fps if fps > 0 else 0.033)
                
        except Exception as e:
            pipeline.logger.log_event("ERROR", f"Stream processing error: {str(e)}")
            raise e
            
        finally:
            # Cleanup
            cap.release()
            self.processing = False
            
            # Log kaydet
            pipeline.logger.log_event("REALTIME_END", "Real-time processing completed")
            log_file = pipeline.logger.save_log()
            
            return {
                'log_file': log_file,
                'metrics': self.metrics,
                'total_frames': frame_count,
                'status': 'completed'
            }
        
    # Tam video işleme ve output_path'e video yazımı yapar 
    def process_video_realtime(self, input_path, output_path, yolo_model_path, unet_model_path):
        """Eski stil video işleme - tam video işleme"""
        try:
            # Pipeline başlat
            pipeline = RunwayDetectionPipeline(yolo_model_path, unet_model_path)
            pipeline.logger.log_event("BATCH_START", f"Batch processing started: {input_path}")
            
            # Video input/output
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise Exception(f"Video açılamadı: {input_path}")
            
            # Video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            processed_frames = 0
            
            # Progress bar oluştur
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Frame işle
                processed_frame = self.process_frame_realtime(frame, pipeline)
                out.write(processed_frame)
                
                frame_count += 1
                processed_frames += 1
                
                # Progress güncelle
                progress = frame_count / total_frames if total_frames > 0 else 0
                progress_bar.progress(progress)
                status_text.text(f"İşlenen frame: {frame_count}/{total_frames}")
                
                # Metrics güncelle
                self.update_metrics_from_pipeline(pipeline)
            
            # Cleanup
            cap.release()
            out.release()
            
            # Log kaydet
            pipeline.logger.log_event("BATCH_END", f"Batch processing completed. Frames: {processed_frames}")
            log_file = pipeline.logger.save_log()
            
            return output_path, log_file, self.metrics
            
        except Exception as e:
            if 'cap' in locals():
                cap.release()
            if 'out' in locals():
                out.release()
            raise e
        
    # İşlemi manuel olarak durdurma 
    def stop_processing_stream(self):
        """İşlemi durdur"""
        self.stop_processing = True
        self.processing = False
    
    # YOLO/U-Net çıktılarından Streamlit arayüzünde gösterilecek metirkler güncellenir 
    def update_metrics_from_pipeline(self, pipeline):
        """Pipeline'dan metrics güncelle"""
        self.metrics.update({
            'runway_detected': pipeline.runway_detected,
            'segmentation_active': pipeline.in_segmentation_mode,
            'confidence': getattr(pipeline, 'last_confidence', 0.0),
            'mode': 'Segmentasyon Aktif' if pipeline.in_segmentation_mode else 
                   ('Pist Tespit Edildi' if pipeline.runway_detected else 'Pist Aranıyor...')
        })
        
        # Deviation hesaplama
        if hasattr(pipeline, 'last_deviation'):
            self.metrics['deviation'] = pipeline.last_deviation

    # İşlem durumu JSON benzeri dict olarak döner 
    def get_processing_status(self):
        """Mevcut işleme durumunu döndür"""
        return {
            'processing': self.processing,
            'metrics': self.metrics.copy(),
            'stop_requested': self.stop_processing
        }
    
# Frame-by-frame işleyip anlık görsel ve metrik çıktısı üretir 
def process_video_realtime_streamlit(input_path, yolo_model_path, unet_model_path, 
                                   video_placeholder, metrics_placeholder, progress_bar):
    """
    Streamlit için optimize edilmiş gerçek zamanlı video işleme
    """
    processor = StreamlitVideoProcessor()
    
    def frame_callback(processed_frame, frame_count, total_frames):
        """Her frame için çağrılacak callback"""
        # Frame'i RGB'ye çevir ve göster
        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        
        with video_placeholder.container():
            st.markdown('<div style="border: 2px solid #28a745; padding: 10px; border-radius: 10px;">', 
                       unsafe_allow_html=True)
            st.image(frame_rgb, channels="RGB", use_column_width=True)
            st.markdown(f"**🎬 Frame:** {frame_count}/{total_frames} | **⚡ Canlı İşleme**")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Progress güncelle
        if total_frames > 0:
            progress = frame_count / total_frames
            progress_bar.progress(progress)
    
    def metrics_callback(metrics):
        """Metrics güncellemesi için callback"""
        with metrics_placeholder.container():
            col1, col2 = st.columns(2)
            
            with col1:
                status = "✅ Tespit Edildi" if metrics['runway_detected'] else "🔍 Aranıyor"
                color = "#28a745" if metrics['runway_detected'] else "#ffc107"
                st.markdown(f'<div style="background-color: {color}20; padding: 0.5rem; border-radius: 5px; border-left: 4px solid {color};">'
                           f'<strong>Pist:</strong> {status}</div>', unsafe_allow_html=True)
            
            with col2:
                seg_status = "🎯 Aktif" if metrics['segmentation_active'] else "⏳ Bekliyor"
                seg_color = "#28a745" if metrics['segmentation_active'] else "#ffc107"
                st.markdown(f'<div style="background-color: {seg_color}20; padding: 0.5rem; border-radius: 5px; border-left: 4px solid {seg_color};">'
                           f'<strong>Segmentasyon:</strong> {seg_status}</div>', unsafe_allow_html=True)
            
            # Güven skoru
            confidence = metrics.get('confidence', 0)
            conf_text = f"{confidence:.1%}" if confidence > 0 else "Bekleniyor"
            st.markdown(f'<div style="background-color: #f0f2f620; padding: 0.5rem; border-radius: 5px; border-left: 4px solid #2a5298;">'
                       f'<strong>Güven Skoru:</strong> {conf_text}</div>', unsafe_allow_html=True)
            
            # Sapma değeri (varsa)
            if metrics.get('deviation', 0) != 0:
                deviation = metrics['deviation']
                dev_color = "#dc3545" if abs(deviation) > 50 else "#28a745"
                st.markdown(f'<div style="background-color: #f0f2f620; padding: 0.5rem; border-radius: 5px; border-left: 4px solid {dev_color};">'
                           f'<strong>Sapma:</strong> {deviation:.1f}°</div>', unsafe_allow_html=True)
    
    try:
        # Stream işleme başlat
        result = processor.process_video_stream(
            input_path, 
            yolo_model_path, 
            unet_model_path,
            frame_callback=frame_callback,
            metrics_callback=metrics_callback
        )
        
        return result
        
    except Exception as e:
        st.error(f"Gerçek zamanlı işleme hatası: {str(e)}")
        return {
            'error': str(e),
            'status': 'failed'
        }
    
# Arka plan işleme için (geri uyumluluk), process_video_realtime çağırır
def process_video(input_path, output_path, yolo_model_path, unet_model_path):
    """Mevcut process_video fonksiyonu - geriye dönük uyumluluk için"""
    processor = StreamlitVideoProcessor()
    
    # İşleme başlat
    with st.spinner('Video işleniyor... Lütfen bekleyin.'):
        try:
            # Eski stil işleme (tüm video işlendikten sonra sonuç)
            output_file, log_file, final_metrics = processor.process_video_realtime(
                input_path, output_path, yolo_model_path, unet_model_path
            )
            
            return output_file, {
                'log_file': log_file,
                'metrics': final_metrics,
                'status': 'completed'
            }
            
        except Exception as e:
            st.error(f"İşlem sırasında hata oluştu: {str(e)}")
            return None, {
                'error': str(e),
                'status': 'failed'
            }
# Videoya veya log dosyalarını base64 formatında indirilebilir hale getirir 
def create_download_link(file_path, link_text="İndir", file_name=None):
    """Dosya indirme linki oluştur"""
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, "rb") as f:
            file_data = f.read()
        
        b64_data = base64.b64encode(file_data).decode()
        
        if file_name is None:
            file_name = os.path.basename(file_path)
        
        return f'<a href="data:application/octet-stream;base64,{b64_data}" download="{file_name}">{link_text}</a>'
    
    except Exception as e:
        st.error(f"İndirme linki oluşturulamadı: {str(e)}")
        return None

# Video dosyasının çözünürlük, FPS, süre ve boyut bilgilerini alır 
def display_video_info(video_path):
    """Video bilgilerini göster"""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            'fps': fps,
            'width': width,
            'height': height,
            'total_frames': total_frames,
            'duration': duration,
            'file_size': os.path.getsize(video_path) if os.path.exists(video_path) else 0
        }
    
    except Exception as e:
        st.error(f"Video bilgileri alınamadı: {str(e)}")
        return None

# YOLO ve U-net model yollarını ve uzantılarını kontrol eder 
def validate_models(yolo_path, unet_path):
    """Model dosyalarını doğrula"""
    errors = []
    
    if not os.path.exists(yolo_path):
        errors.append(f"YOLO model dosyası bulunamadı: {yolo_path}")
    
    if not os.path.exists(unet_path):
        errors.append(f"U-Net model dosyası bulunamadı: {unet_path}")
    
    # Dosya uzantılarını kontrol et
    if yolo_path and not yolo_path.lower().endswith(('.pt', '.onnx', '.engine')):
        errors.append("YOLO model dosyası geçersiz format (.pt, .onnx, .engine desteklenir)")
    
    if unet_path and not unet_path.lower().endswith(('.pt', '.pth', '.onnx')):
        errors.append("U-Net model dosyası geçersiz format (.pt, .pth, .onnx desteklenir)")
    
    return errors

# Temp dosyalarını siler 
def cleanup_temp_files(file_paths):
    """Geçici dosyaları temizle"""
    cleaned_files = []
    
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                cleaned_files.append(file_path)
        except Exception as e:
            st.warning(f"Geçici dosya silinemedi {file_path}: {str(e)}")
    
    return cleaned_files

# Byte türünden dosya boyutunu okunabilir birimlere dönüştürür
def format_file_size(size_bytes):
    """Dosya boyutunu okunabilir formata çevir"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"

# Saniyeyi saat, dakika, saniye formatına dönüştürür
def format_duration(seconds):
    """Süreyi okunabilir formata çevir"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

"""
# st.session_state içindeki gerekli anahtarları ilk defa başlatır 
def init_session_state():
    #Session state'i başlat
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    
    if 'last_result' not in st.session_state:
        st.session_state.last_result = None
    
    if 'processor' not in st.session_state:
        st.session_state.processor = None
"""

# işlem durduğunda veya yeniden başlatıldığında session_state'i temizler 
def reset_session_state():
    """Session state'i sıfırla"""
    st.session_state.processing = False
    st.session_state.last_result = None
    if st.session_state.processor:
        st.session_state.processor.stop_processing_stream()
    st.session_state.processor = None
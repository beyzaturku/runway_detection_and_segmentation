import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import os
import tempfile
from datetime import datetime
from ultralytics import YOLO
from PIL import Image
import base64
import sys 
import threading
import time
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'codes')))
from processing_video import process_video
from session_utils import init_session_state

# Sayfa yapılandırma ve CSS ayarlarını başlatır 
class UIBuilder:
    def __init__(self):
        init_session_state()
        self.setup_page()
        self.set_custom_css()

    def setup_page(self):
        pass

    def set_custom_css(self):
        st.markdown("""<style>
            .main-header {
                background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
                padding: 1rem;
                border-radius: 10px;
                color: white;
                text-align: center;
                margin-bottom: 2rem;
            }
            .video-box {
                border: 2px dashed #2a5298;
                padding: 20px;
                min-height: 400px;
                background-color: #f9f9f9;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 10px;
                text-align: center;
            }
            .metric-card {
                background: #f0f2f6;
                padding: 1rem;
                border-radius: 8px;
                border-left: 4px solid #2a5298;
                margin: 0.5rem 0;
            }
            .status-success {
                background: #d4edda;
                color: #155724;
                padding: 0.75rem;
                border-radius: 5px;
                border-left: 4px solid #28a745;
                margin: 0.5rem 0;
            }
            .status-warning {
                background: #fff3cd;
                color: #856404;
                padding: 0.75rem;
                border-radius: 5px;
                border-left: 4px solid #ffc107;
                margin: 0.5rem 0;
            }
            .report-section {
                background: #e7f3ff;
                padding: 1rem;
                border-radius: 8px;
                border-left: 4px solid #007bff;
                margin: 1rem 0;
            }
            .download-btn {
                background: #28a745;
                color: white;
                padding: 0.5rem 1rem;
                border-radius: 5px;
                text-decoration: none;
                display: inline-block;
                margin: 0.5rem 0;
            }
        </style>""", unsafe_allow_html=True)

    # Sayfanın üst kısmında sistemin başlığını ve açıklamasını gösterir 
    def render_header(self):
        st.markdown("""
        <div class="main-header">
            <h1>🛬 Gerçek Zamanlı Pist Takip Sistemi</h1>
            <p>YOLO + U-Net + Polinom Sapma Analizi - Canlı İşleme</p>
        </div>
        """, unsafe_allow_html=True)
   
    # CSV raporu oluşturma 
    def create_processing_report(self):
        """İşleme raporunu CSV formatında oluştur"""
        if not hasattr(st.session_state, 'processing_log') or not st.session_state.processing_log:
            return None
            
        # Rapor verilerini DataFrame'e dönüştür
        df = pd.DataFrame(st.session_state.processing_log)
        
        # Zaman bilgisini ekle (saniye cinsinden)
        fps = st.session_state.current_metrics.get('fps', 30)
        df['time_seconds'] = df['frame_number'] / fps
        df['time_seconds'] = df['time_seconds'].round(2)
        
        # Sütun sırasını düzenle
        df = df[['frame_number', 'time_seconds', 'runway_detected', 'segmentation_active', 'confidence', 'processing_time']]
        
        # CSV dosyasını kaydet
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"runway_detection_report_{timestamp}.csv"
        csv_path = os.path.join(tempfile.gettempdir(), csv_filename)
        
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        return csv_path, df

    # YOLO + U-net ile gerçek zamanlı frame-by-frame video işleme yapar, metrikleri günceller 
    def process_video_realtime(self, video_path, yolo_model_path, unet_model_path, video_placeholder, metrics_placeholder):
        from codes.main_pipeline import RunwayDetectionPipeline
        pipeline = RunwayDetectionPipeline(yolo_model_path, unet_model_path)

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        st.session_state.total_frames = total_frames
        st.session_state.current_metrics['fps'] = fps
        st.session_state.processing_log = []  # İşleme logunu başlat
        st.session_state.processed_frames = []  # İşlenmiş frame'leri sakla

         # Önceki indirme verilerini temizle
        if hasattr(st.session_state, 'video_bytes'):
            delattr(st.session_state, 'video_bytes')
        if hasattr(st.session_state, 'report_csv'):
            delattr(st.session_state, 'report_csv')
        if hasattr(st.session_state, 'report_summary'):
            delattr(st.session_state, 'report_summary')

        progress_bar = st.progress(0)
        frame_count = 0
        start_time = time.time()

        while st.session_state.processing and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_start_time = time.time()
            processed_frame = pipeline.process_frame(frame)
            frame_processing_time = time.time() - frame_start_time

            # Metrikleri güncelle
            st.session_state.current_metrics.update({
                'runway_detected': pipeline.runway_detected,
                'segmentation_active': pipeline.in_segmentation_mode,
                'confidence': getattr(pipeline, 'last_confidence', 0.0),
                'frame_number': frame_count,
                'mode': 'Segmentasyon Aktif' if pipeline.in_segmentation_mode else ('Pist Tespit Edildi' if pipeline.runway_detected else 'Pist Aranıyor...')
            })

            # İşleme loguna kaydet
            log_entry = {
                'frame_number': frame_count,
                'runway_detected': pipeline.runway_detected,
                'segmentation_active': pipeline.in_segmentation_mode,
                'confidence': getattr(pipeline, 'last_confidence', 0.0),
                'processing_time': round(frame_processing_time, 4)
            }
            st.session_state.processing_log.append(log_entry)

            # İşlenmiş frame'i sakla
            st.session_state.processed_frames.append(processed_frame)

            # Frame'i göster
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB")

            # Metrikleri güncelle
            with metrics_placeholder.container():
                self.render_realtime_metrics()

            # İlerleme çubuğunu güncelle
            progress = (frame_count + 1) / total_frames
            progress_bar.progress(progress)

            frame_count += 1
            st.session_state.frame_count = frame_count
            
            # FPS'e göre bekleme süresi
            time.sleep(0.01)  # Daha hızlı işleme için kısaltıldı

        cap.release()
        
        # İşlem tamamlandığında
        if frame_count >= total_frames or not st.session_state.processing:
            st.session_state.processing = False
            st.session_state.processing_complete = True
            st.session_state.total_processing_time = time.time() - start_time
            progress_bar.progress(1.0)
            
            # Video kaydetme için hazırla
            if st.session_state.processed_frames:
                st.session_state.video_ready_to_save = True
    
    # Canlı işleme sırasında tespit, segmentasyon, güven ve frame durumu gibi bilgileri gösterir.
    def render_realtime_metrics(self):
        metrics = st.session_state.current_metrics

        # Pist durumu
        runway_status = "✅ Tespit Edildi" if metrics.get("runway_detected") else "🔍 Aranıyor"
        st.markdown(f'<div class="status-{"success" if metrics.get("runway_detected") else "warning"}"><strong>Pist:</strong> {runway_status}</div>', unsafe_allow_html=True)

        # Segmentasyon durumu
        seg_status = "🎯 Aktif" if metrics.get("segmentation_active") else "⏳ Bekliyor"
        st.markdown(f'<div class="status-{"success" if metrics.get("segmentation_active") else "warning"}"><strong>Segmentasyon:</strong> {seg_status}</div>', unsafe_allow_html=True)

        # Frame bilgisi
        frame_info = f"{metrics.get('frame_number', 0)}/{st.session_state.total_frames}"
        st.markdown(f'<div class="metric-card"><strong>Frame:</strong> {frame_info}</div>', unsafe_allow_html=True)

        # FPS bilgisi
        fps_info = metrics.get('fps', 0)
        st.markdown(f'<div class="metric-card"><strong>FPS:</strong> {fps_info}</div>', unsafe_allow_html=True)

        # İşlem süresi (canlı)
        if st.session_state.processing and hasattr(st.session_state, 'processing_log') and st.session_state.processing_log:
            avg_time = np.mean([log['processing_time'] for log in st.session_state.processing_log])
            st.markdown(f'<div class="metric-card"><strong>Ort. İşlem Süresi:</strong> {avg_time:.3f}s</div>', unsafe_allow_html=True)

    # Video kaydetme 
    def save_processed_video(self, output_path):
        """İşlenmiş videoyu kaydet"""
        if not hasattr(st.session_state, 'processed_frames') or not st.session_state.processed_frames:
            return False
            
        try:
            # Video yazıcıyı başlat
            height, width, _ = st.session_state.processed_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = st.session_state.current_metrics.get('fps', 30)
            
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Frame'leri kaydet
            for frame in st.session_state.processed_frames:
                out.write(frame)
                
            out.release()
            return True
        except Exception as e:
            st.error(f"Video kaydetme hatası: {str(e)}")
            return False
  
    # Sayfanın ana içeriğini ve kullanıcı arayüzünü oluşturur: video yükleme, test başlatma, durdurma, sıfırlama gibi. 
    def render_main_layout(self):
        st.subheader("📹 Canlı Video İşleme Paneli")
        col_video, col_metrics = st.columns([3, 1])
        #video_placeholder = col_video.empty()
        metrics_placeholder = col_metrics.empty()

        with col_video:
            video_placeholder = st.empty()

        with col_metrics:
            st.subheader("📊 Canlı Durum Paneli")
            metrics_placeholder = st.empty()
          
            # İlerleme bilgisi
            if st.session_state.processing and st.session_state.frame_count > 0:
                percent = (st.session_state.frame_count / st.session_state.total_frames) * 100
                st.markdown(f'<div class="status-success"><strong>İlerleme:</strong> %{percent:.1f}</div>', unsafe_allow_html=True)

        with col_metrics:
            with metrics_placeholder.container():
                self.render_realtime_metrics()

        st.markdown("### 🎛️ Kontrol Paneli")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            uploaded_file = st.file_uploader("📁 Video Yükle", type=["mp4", "avi", "mov", "mkv"], help="200MB'a kadar desteklenir")
            if uploaded_file and not st.session_state.processing:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp:
                    temp.write(uploaded_file.read())
                    st.session_state.video_path = temp.name
                    st.session_state.video_uploaded = True
                    st.session_state.processing_complete = False
                    st.session_state.video_ready_to_save = False
                    st.session_state.show_download_buttons = False

                st.success("✅ Video başarıyla yüklendi!")
                st.info(f"📊 Dosya boyutu: {len(uploaded_file.getvalue()) / (1024*1024):.1f} MB")

                cap = cv2.VideoCapture(st.session_state.video_path)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, channels="RGB")
        with col2:
            if st.button("🚀 Canlı Test Başlat", use_container_width=True, 
                        disabled=not st.session_state.video_uploaded or st.session_state.processing):
                st.session_state.processing = True
                st.session_state.processing_complete = False
                st.session_state.frame_count = 0
                st.session_state.video_ready_to_save = False
                st.session_state.show_download_buttons = False

                BASE_DIR = os.path.dirname(os.path.abspath(__file__))
                yolo_weights = os.path.join(BASE_DIR, "..", "trained_models", "best.pt")
                yolo_weights = os.path.abspath(yolo_weights)

                unet_weights = os.path.join(BASE_DIR, "..", "trained_models", "unet_final.pth")
                unet_weights = os.path.abspath(unet_weights)

                try:
                    self.process_video_realtime(
                        st.session_state.video_path,
                        yolo_weights,
                        unet_weights,
                        video_placeholder,
                        metrics_placeholder
                    )
                except Exception as e:
                    st.error(f"Hata: {str(e)}")
                    st.session_state.processing = False

        with col3:
            if st.button("💾 Kaydet", use_container_width=True, 
                disabled=not (st.session_state.processing_complete and 
                    getattr(st.session_state, 'video_ready_to_save', True))):
        
                # Kalıcı indirme durumunu ayarla
                st.session_state.show_download_buttons = True
                st.session_state.download_session_active = True
        
                # Download session ID'yi oluştur (eğer yoksa)
                if 'download_session_id' not in st.session_state:
                    st.session_state.download_session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                
                output_filename = f"processed_runway_video_{st.session_state.download_session_id}.mp4"
                output_path = os.path.join(tempfile.gettempdir(), output_filename)

                if hasattr(st.session_state, 'processed_frames') and st.session_state.processed_frames:
                    if self.save_processed_video(output_path):
                        with open(output_path, "rb") as f:
                            st.session_state.video_bytes = f.read()
                        if os.path.exists(output_path):
                            os.remove(output_path)
                
                if hasattr(st.session_state, 'processing_log') and st.session_state.processing_log:
                    report_data = self.create_processing_report()
                    if report_data:
                        csv_path, df = report_data
                        with open(csv_path, "r", encoding='utf-8') as f:
                            st.session_state.report_csv = f.read()
                        if os.path.exists(csv_path):
                            os.remove(csv_path)
                        st.session_state.report_summary = {
                            'total_frames': len(df),
                            'runway_detected_frames': df['runway_detected'].sum(),
                            'segmentation_active_frames': df['segmentation_active'].sum(),
                            'avg_confidence': df[df['confidence'] > 0]['confidence'].mean() if df['confidence'].sum() > 0 else 0,
                            'total_time': df['time_seconds'].max() if not df.empty else 0,
                            'avg_processing_time': df['processing_time'].mean()
                        }
        
                # Verileri hazırla
                #self.prepare_download_data()
        
                st.success("✅ Video hazır! Aşağıdan indirin:")

        with col4:
            if st.button("🔄 Sıfırla", use_container_width=True, disabled=st.session_state.processing):
            # Tüm download session verilerini temizle
                keys_to_remove = [
                    'video_bytes', 'report_csv', 'report_summary', 
                    'download_session_id', 'download_session_active',
                    'show_download_buttons'
                ]
                
                for key in keys_to_remove:
                    if key in st.session_state:
                        delattr(st.session_state, key)
        
                from session_utils import reset_session_state
                reset_session_state()
                st.success("✅ Sistem sıfırlandı!")
                st.rerun()
        
        if st.session_state.get('show_download_buttons', False):
            self.render_download_buttons_inline()

    def render_download_buttons_inline(self):
        """Kaydet butonunun hemen altında indirme butonlarını göster"""
        if not (getattr(st.session_state, 'show_download_buttons', False) or 
            hasattr(st.session_state, 'download_session_id')):
            return

        st.markdown("### 📥 İndirme Seçenekleri")
    
        self.prepare_download_data()

        # Video indirme alanı
        self.render_video_download_button()
    
        # Rapor indirme alanı  
        self.render_report_download_button()
    
    def render_video_download_button(self):
        #Video indirme butonu
        if hasattr(st.session_state, 'video_bytes'):
            if 'download_session_id' not in st.session_state:
                st.session_state.download_session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
            output_filename = f"processed_runway_video_{st.session_state.download_session_id}.mp4"
        
            # Video boyutu bilgisi
            video_size_mb = len(st.session_state.video_bytes) / (1024 * 1024)
            st.info(f"🎥 Video hazır! Boyut: {video_size_mb:.1f} MB")
        
            st.download_button(
                label="🎥 İşlenmiş Videoyu İndir",
                data=st.session_state.video_bytes,
                file_name=output_filename,
                mime="video/mp4",
                use_container_width=True,
                key=f"persistent_video_btn_{st.session_state.download_session_id}",
                help="Videoyu bilgisayarınıza indirin"
            )
        else:
            st.error("❌ Kaydedilecek video bulunamadı!")

    def render_report_download_button(self):
        #Rapor indirme butonu ve metrikleri
        if hasattr(st.session_state, 'report_csv'):
            if not hasattr(st.session_state, 'download_session_id'):
                st.session_state.download_session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            
            csv_filename = f"runway_detection_report_{st.session_state.download_session_id}.csv"
        
            # Rapor boyutu bilgisi
            report_size_kb = len(st.session_state.report_csv.encode('utf-8')) / 1024
            st.info(f"📊 Rapor hazır! Boyut: {report_size_kb:.1f} KB")
        
            # Download butonu
            st.download_button(
                label="📊 Rapor İndir (CSV)",
                data=st.session_state.report_csv,
                file_name=csv_filename,
                mime="text/csv",
                use_container_width=True,
                key=f"persistent_report_btn_{st.session_state.download_session_id}",
                help="Detaylı analiz raporunu indirin"
            )

            # Rapor özetini her zaman göster
            self.render_report_summary()
     
    def render_report_summary(self):
        """Rapor özetini göster - her zaman görünür"""
        if hasattr(st.session_state, 'report_summary'):
            summary = st.session_state.report_summary
        
            st.markdown("#### 📈 Rapor Özeti")
        
            # Metrikler her zaman görünür
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Toplam Frame", f"{summary['total_frames']}")
                st.metric("Pist Tespit Edilen Frame", f"{summary['runway_detected_frames']}")
                st.metric("Segmentasyon Aktif Frame", f"{summary['segmentation_active_frames']}")
            
            with col2:
                st.metric("Ortalama Güven", f"{summary['avg_confidence']:.1%}" if summary['avg_confidence'] > 0 else "N/A")
                st.metric("Toplam Video Süresi", f"{summary['total_time']:.2f} saniye")
                st.metric("Ort. İşlem Süresi/Frame", f"{summary['avg_processing_time']:.3f} saniye")
        
            # Özet istatistik
            detection_rate = (summary['runway_detected_frames'] / summary['total_frames']) * 100
            segmentation_rate = (summary['segmentation_active_frames'] / summary['total_frames']) * 100
        
            st.markdown("#### 🎯 Performans Özeti")
            perf_col1, perf_col2 = st.columns(2)
            with perf_col1:
                st.metric("Pist Tespit Oranı", f"%{detection_rate:.1f}")
            with perf_col2:
                st.metric("Segmentasyon Oranı", f"%{segmentation_rate:.1f}")

    def prepare_download_data(self):
        #Video ve rapor verilerini session state'e hazırla
        if 'download_session_id' not in st.session_state:
            st.session_state.download_session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    
        output_filename = f"processed_runway_video_{st.session_state.download_session_id}.mp4"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)

        if 'video_bytes' not in st.session_state:
            if hasattr(st.session_state, 'processed_frames') and st.session_state.processed_frames:
                if self.save_processed_video(output_path):
                    with open(output_path, "rb") as f:
                        st.session_state.video_bytes = f.read()
                    # Temp dosyayı temizle
                    if os.path.exists(output_path):
                        os.remove(output_path)

        if 'report_csv' not in st.session_state:
            if hasattr(st.session_state, 'processing_log') and st.session_state.processing_log:
                report_data = self.create_processing_report()
                if report_data:
                    csv_path, df = report_data
                    with open(csv_path, "r", encoding='utf-8') as f:
                        st.session_state.report_csv = f.read()
                    # Temp dosyayı temizle
                    if os.path.exists(csv_path):
                        os.remove(csv_path)

                    # Rapor özetini kaydet
                    st.session_state.report_summary = {
                        'total_frames': len(df),
                        'runway_detected_frames': df['runway_detected'].sum(),
                        'segmentation_active_frames': df['segmentation_active'].sum(),
                        'avg_confidence': df[df['confidence'] > 0]['confidence'].mean() if df['confidence'].sum() > 0 else 0,
                        'total_time': df['time_seconds'].max() if not df.empty else 0,
                        'avg_processing_time': df['processing_time'].mean()
                    }

    def render_footer(self):
        st.markdown("---")
        if st.session_state.processing:
            msg = f"🔄 Canlı İşlem - Frame: {st.session_state.frame_count}/{st.session_state.total_frames}"
        elif st.session_state.processing_complete:
            msg = "✅ İşlem Tamamlandı - Video ve Rapor Hazır"
        elif st.session_state.video_uploaded:
            msg = "📹 Video Yüklendi - Test Edilmeye Hazır"
        else:
            msg = "🔧 Sistem Hazır"
        
        st.markdown(f"**Durum:** {msg}")
        
        # Ek bilgiler
        if hasattr(st.session_state, 'total_processing_time'):
            st.markdown(f"**Toplam İşlem Süresi:** {st.session_state.total_processing_time:.2f} saniye")


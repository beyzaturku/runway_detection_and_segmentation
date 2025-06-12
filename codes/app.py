import streamlit as st
import tempfile
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ui.layout import UIBuilder
from session_utils import init_session_state

# Sayfa başlığı
st.set_page_config(
    page_title="🛬 Pist Takip Sistemi",
    layout="wide",
    initial_sidebar_state="expanded"
)

def cleanup_temp_files():
    """Geçici dosyaları temizle"""
    try:
        if hasattr(st.session_state, 'video_path') and st.session_state.video_path:
            if os.path.exists(st.session_state.video_path):
                os.unlink(st.session_state.video_path)
    except Exception as e:
        st.sidebar.warning(f"Geçici dosya temizleme uyarısı: {str(e)}")

def sidebar_info():
    """Yan panel bilgileri"""
    st.sidebar.title("📊 Sistem Bilgileri")
    
    # Genel Durum
    st.sidebar.subheader("🔧 Sistem Durumu")
    
    if st.session_state.processing:
        st.sidebar.info("⚙️ Video işleniyor...")
    elif st.session_state.video_uploaded:
        st.sidebar.success("✅ Video yüklendi")
    else:
        st.sidebar.warning("📁 Video bekleniyor")
    
    # Model Bilgileri
    st.sidebar.subheader("🤖 Model Bilgileri")
    st.sidebar.info("""
    **YOLO Model:** Pist Tespiti
    **U-Net Model:** Pist Segmentasyonu
    **Tracking:** Kalman Filtresi
    **Smoothing:** Temporal Averaging
    """)
    
    # İstatistikler
    if hasattr(st.session_state, 'current_metrics'):
        st.sidebar.subheader("📈 Mevcut Durum")
        metrics = st.session_state.current_metrics
        
        # Detection durumu
        if metrics.get('runway_detected', False):
            st.sidebar.success("🎯 Pist Tespit Edildi")
        else:
            st.sidebar.warning("🔍 Pist Aranıyor")
        
        # Segmentation durumu
        if metrics.get('segmentation_active', False):
            st.sidebar.success("🛬 Segmentasyon Aktif")
        else:
            st.sidebar.info("⏳ Segmentasyon Bekliyor")
        
        # Confidence
        confidence = metrics.get('confidence', 0)
        if confidence > 0:
            st.sidebar.metric("Güven Skoru", f"{confidence:.1%}")
    
    # Temizlik butonu
    st.sidebar.subheader("🧹 Sistem Temizliği")
    if st.sidebar.button("🗑️ Geçici Dosyaları Temizle"):
        cleanup_temp_files()
        st.sidebar.success("Temizlendi!")
    
    # Yardım
    st.sidebar.subheader("❓ Yardım")
    with st.sidebar.expander("Kullanım Kılavuzu"):
        st.write("""
        **1. Video Yükleme:**
        - MP4, AVI, MOV, MKV formatları desteklenir
        - Maksimum 200MB boyut limiti
        
        **2. Test Etme:**
        - Video yüklendikten sonra 'Test Et' butonuna tıklayın
        - İşlem sırasında ilerleme takip edilebilir
        
        **3. Sonuçları Kaydetme:**
        - İşlenmiş videoyu MP4 formatında indirebilirsiniz
        - Detaylı raporu CSV formatında alabilirsiniz
        
        **4. Durum Takibi:**
        - Sağ panelden anlık durumu takip edebilirsiniz
        - Yan panelden sistem bilgilerini görebilirsiniz
        """)

def main():
    """Ana uygulama fonksiyonu"""
    #from session_utils import init_session_state
    init_session_state()

    # Ana UI
    ui = UIBuilder()
    ui.render_header()
    
    # Yan panel
    #sidebar_info()
    
    # Ana layout
    ui.render_main_layout()
    
    # Footer
    ui.render_footer()

if __name__ == "__main__":
    main()



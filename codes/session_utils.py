import streamlit as st

def init_session_state():
    """Tüm gerekli session_state anahtarlarını başlatır."""
    defaults = {
        "video_uploaded": False,
        "video_path": None,
        "processing": False,
        "output_video_path": None,
        "log_file_path": None,
        "current_metrics": {
            'runway_detected': False,
            'segmentation_active': False,
            'confidence': 0.0,
            'mode': 'Hazır',
            'frame_number': 0,
            'fps': 0
        },
        "frame_count": 0,
        "total_frames": 0,
        "processing_complete": False,
        "processor": None
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_session_state():
    """Tüm session_state değerlerini sıfırlar."""
    keys_to_reset = [
        "video_uploaded", "video_path", "processing",
        "output_video_path", "log_file_path", "processing_complete",
        "frame_count", "total_frames", "processor"
    ]

    for key in keys_to_reset:
        st.session_state[key] = False if key != "video_path" else None

    st.session_state["current_metrics"] = {
        'runway_detected': False,
        'segmentation_active': False,
        'confidence': 0.0,
        'mode': 'Hazır',
        'frame_number': 0,
        'fps': 0
    }

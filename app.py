import os
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# Import library Grad-CAM++
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# ======================================================================
# CONFIGURATION & LOAD MODEL
# ======================================================================
st.set_page_config(page_title="OnionCare AI", layout="wide")

# Jalur bobot model (sesuaikan dengan file lokal kamu, atau buat fallback otomatis)
WEIGHTS_PATH = r"mobilenet_v2_bawang.pth" # Menggunakan nama lokal di folder yang sama agar fleksibel
class_names = ['Bercak_Ungu', 'Busuk_Daun', 'Fusarium', 'Sehat']
scientific_names = {
    'Bercak_Ungu': 'Alternaria porri',
    'Busuk_Daun': 'Peronospora destructor',
    'Fusarium': 'Fusarium oxysporum',
    'Sehat': 'Allium ascalonicum (Sehat)'
}
num_classes = len(class_names)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_onion_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    if os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    target_layers = [model.features[-1]]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    return model, cam

try:
    model, cam = load_onion_model()
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file pth ada di {WEIGHTS_PATH}")

# ======================================================================
# REVISI TRANSFORMASI: Menggunakan Resize(256) & CenterCrop(224)
# ======================================================================
data_transform = transforms.Compose([
    transforms.Resize(256),           
    transforms.CenterCrop(224),      
    transforms.ToTensor(),
])
normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

# ======================================================================
# INJEKSI CSS CUSTOM (Biar UI Mirip Mockup: Rounded Card, Soft Shadow)
# ======================================================================
st.markdown("""
    <style>
    .main { background-color: #F8F9FC; }
    .title-container { margin-bottom: 25px; }
    .title-main { font-size: 28px; font-weight: 700; color: #1E293B; margin-bottom: 5px; }
    .subtitle-main { font-size: 14px; color: #64748B; }
    
    .custom-card {
        background-color: #FFFFFF;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.04);
        border: 1px solid #E2E8F0;
        margin-bottom: 20px;
    }
    .card-title { font-size: 16px; font-weight: 600; color: #1E293B; margin-bottom: 15px; display: flex; align-items: center; }
    
    .badge-danger { background-color: #FFEEF3; color: #F1416C; padding: 4px 12px; border-radius: 6px; font-size: 12px; font-weight: 600; }
    .badge-success { background-color: #E8FFF3; color: #50CD89; padding: 4px 12px; border-radius: 6px; font-size: 12px; font-weight: 600; }
    
    .stProgress > div > div > div > div { background-color: #7239EA; }
    </style>
""", unsafe_allow_html=True)

# ======================================================================
# HEADER WEBSITE
# ======================================================================
st.markdown("""
    <div class="title-container">
        <div class="title-main">Selamat datang di OnionCare AI 👋</div>
        <div class="subtitle-main">Upload gambar daun bawang merah untuk mendeteksi penyakit secara cepat dan akurat.</div>
    </div>
""", unsafe_allow_html=True)

col_top_left, col_top_right = st.columns([1.8, 1])

with col_top_left:
    st.markdown('<div class="custom-card"><div class="card-title">1. Upload Gambar Daun Bawang Merah</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drag & Drop gambar di sini atau klik pilih gambar", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

with col_top_right:
    st.markdown("""
        <div class="custom-card">
            <div class="card-title">💡 Tips Foto yang Baik</div>
            <table style="width:100%; border-collapse: collapse; font-size: 13px; color: #475569;">
                <tr>
                    <td style="padding: 8px 0; font-weight:500;">📍 Gunakan pencahayaan cukup</td>
                    <td style="padding: 8px 0; font-weight:500;">📸 Foto bagian daun bergejala</td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; font-weight:500;">🔍 Pastikan gambar fokus</td>
                    <td style="padding: 8px 0; font-weight:500;">🌤️ Hindari bayangan pada daun</td>
                </tr>
            </table>
        </div>
    """, unsafe_allow_html=True)

# Bagian Panduan Foto (Menggunakan pengecekan file agar tidak error FileNotFound jika gambar belum ada)
st.subheader("📸 Cara Mengambil Foto yang Benar")
st.write("Pastikan foto daun bawangmu memenuhi kriteria berikut agar hasil deteksi akurat:")

col1, col2, col3, col4 = st.columns(4)

with col1:
    if os.path.exists("contoh1.jpeg"):
        st.image("contoh1.jpeg", caption="Pencahayaan Terang", use_container_width=True)
    else:
        st.info("Pencahayaan Terang\n*(Letakkan contoh1.jpeg)*")
with col2:
    if os.path.exists("contoh2.jpg"):
        st.image("contoh2.jpg", caption="Fokus ke Daun", use_container_width=True)
    else:
        st.info("Fokus ke Daun\n*(Letakkan contoh2.jpg)*")
with col3:
    if os.path.exists("contoh3.jpg"):
        st.image("contoh3.jpg", caption="Background Polos", use_container_width=True)
    else:
        st.info("Background Polos\n*(Letakkan contoh3.jpg)*")
with col4:
    if os.path.exists("contoh4.jpg"):
        st.image("contoh4.jpg", caption="Sudut (Angle) Tegak", use_container_width=True)
    else:
        st.info("Sudut Tegak\n*(Letakkan contoh4.jpg)*")

st.markdown("---")

# ======================================================================
# PROSES DETEKSI & TAMPILAN HASIL
# ======================================================================
if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(image_pil)
    
    input_tensor = data_transform(image_pil)
    input_normalized = normalize_transform(input_tensor).to(device).unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_normalized)
        probabilities = F.softmax(output, dim=1)[0]
        conf_score, pred_idx = torch.max(probabilities, 0)
        
    nama_penyakit = class_names[pred_idx.item()]
    nama_latin = scientific_names[nama_penyakit]
    persentase_conf = conf_score.item() * 100
    
    targets = [ClassifierOutputTarget(pred_idx.item())]
    grayscale_cam = cam(input_tensor=input_normalized, targets=targets)[0, :]
    
    img_bg = cv2.resize(img_array, (224, 224))
    img_bg = img_bg.astype(np.float32) / 255.0
    cam_image = show_cam_on_image(img_bg, grayscale_cam, use_rgb=True)
    
    st.markdown('<div class="custom-card"><div class="card-title">🎯 Hasil Deteksi</div>', unsafe_allow_html=True)
    
    col_bottom_left, col_bottom_right = st.columns([1.2, 1])
    
    with col_bottom_left:
        col_img_asli, col_img_cam = st.columns(2)
        
        img_original_resized = Image.fromarray((img_bg * 255).astype(np.uint8))
        img_cam_pil = Image.fromarray(cam_image)
        
        with col_img_asli:
            st.markdown('<div style="text-align: center; font-size: 13px; font-weight: 600; color: #475569; margin-bottom: 8px;">Foto Asli</div>', unsafe_allow_html=True)
            st.image(img_original_resized, use_container_width=True)
            
        with col_img_cam:
            st.markdown('<div style="text-align: center; font-size: 13px; font-weight: 600; color: #475569; margin-bottom: 8px;">Heatmap (GradCAM++)</div>', unsafe_allow_html=True)
            st.image(img_cam_pil, use_container_width=True)
            
    with col_bottom_right:
        badge_status = '<span class="badge-danger">TERDETEKSI</span>' if nama_penyakit != 'Sehat' else '<span class="badge-success">NORMAL</span>'
        tingkat_risiko = 'Tinggi' if nama_penyakit != 'Sehat' else 'Tidak Ada'
        warna_risiko = '#F1416C' if nama_penyakit != 'Sehat' else '#50CD89'
        
        rekomendasi = (
            "Lakukan pengendalian segera dengan fungisida berbahan aktif mankozeb atau tembaga hidroksida, "
            "serta pangkas daun yang bergejala parah untuk mencegah penyebaran infeksi lebih luas."
            if nama_penyakit != 'Sehat' else 
            "Pertahankan sanitasi lahan dengan baik, lakukan penyiraman berkala, dan berikan pupuk berimbang agar imunitas tanaman tetap terjaga."
        )

        st.markdown(f"""
            <div style="font-size: 13px; color: #64748B; margin-bottom: 2px;">Penyakit Terdeteksi {badge_status}</div>
            <div style="font-size: 24px; font-weight: 700; color: #1E293B; margin-bottom: 2px;">{nama_penyakit.replace('_', ' ')}</div>
            <div style="font-size: 13px; font-style: italic; color: #64748B; margin-bottom: 15px;">({nama_latin})</div>
            
            <div style="font-size: 13px; color: #64748B; margin-bottom: 2px;">Confidence</div>
            <div style="font-size: 20px; font-weight: 700; color: #7239EA; margin-bottom: 5px;">{persentase_conf:.1f}%</div>
        """, unsafe_allow_html=True)
        
        st.progress(persentase_conf / 100.0)
        
        st.markdown(f"""
            <div style="margin-top: 15px; font-size: 13px; color: #64748B;">Tingkat Risiko: <span style="color:{warna_risiko}; font-weight:600;">⚠️ {tingkat_risiko}</span></div>
            
            <div style="margin-top: 15px; background-color: #F1FAFF; border-left: 4px solid #009EF7; padding: 12px; border-radius: 4px;">
                <div style="font-size: 13px; font-weight: 600; color: #009EF7; margin-bottom: 4px;">🍃 Rekomendasi Singkat</div>
                <div style="font-size: 12px; color: #475569; line-height: 1.5;">{rekomendasi}</div>
            </div>
        """, unsafe_allow_html=True)
        
    st.markdown('</div>', unsafe_allow_html=True)

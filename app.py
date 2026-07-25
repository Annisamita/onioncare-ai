import os
import gdown
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2

# ======================================================================
# CONFIGURATION & PAGE SETUP
# ======================================================================
st.set_page_config(
    page_title="OnionCare AI - Deteksi Penyakit Bawang Merah",
    page_icon="🧅",
    layout="wide"
)

# ID Google Drive dari file .pth kamu
FILE_ID = "1vExdfcOpo8vLeb0QD4gs6u7KZf2kWdEm"
WEIGHTS_PATH = "densenet121_bawang.pth"  # Diperbarui agar sesuai dengan arsitektur DenseNet121

class_names = ['Bercak_Ungu', 'Busuk_Daun', 'Fusarium', 'Sehat']
scientific_names = {
    'Bercak_Ungu': 'Alternaria porri',
    'Busuk_Daun': 'Peronospora destructor',
    'Fusarium': 'Fusarium oxysporum',
    'Sehat': 'Allium ascalonicum (Sehat)'
}
num_classes = len(class_names)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======================================================================
# LOAD MODEL & GRAD-CAM (DENGAN CACHE & GDOWN)
# ======================================================================
@st.cache_resource
def load_onion_model():
    # Otomatis download file .pth dari Google Drive jika belum ada
    if not os.path.exists(WEIGHTS_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, WEIGHTS_PATH, quiet=False)

    # Inisialisasi arsitektur DenseNet121
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    
    if os.path.exists(WEIGHTS_PATH):
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    
    model = model.to(device)
    model.eval()
    
    # Inisialisasi Grad-CAM++ untuk DenseNet (target layer menggunakan denseblock terakhir)
    from pytorch_grad_cam import GradCAMPlusPlus
    target_layers = [model.features.denseblock4]
    cam = GradCAMPlusPlus(model=model, target_layers=target_layers)
    return model, cam

try:
    model, cam = load_onion_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

# ======================================================================
# PREPROCESSING & TRANSFORM
# ======================================================================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

# ======================================================================
# TAMPILAN ANTARMUKA (UI) STREAMLIT
# ======================================================================
st.title("🧅 OnionCare AI")
st.subheader("Sistem Deteksi Penyakit Daun Bawang Merah & Visualisasi XAI")
st.markdown("---")

uploaded_file = st.file_uploader(
    "Upload gambar daun bawang merah...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Tampilkan gambar asli
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Gambar Daun Bawang Merah (Input)", use_container_width=True)

    # Preprocessing untuk model
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (224, 224))
    tensor_img = transform(image).unsqueeze(0).to(device)

    # Prediksi Model
    with torch.no_grad():
        outputs = model(tensor_img)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
        
    predicted_class = class_names[predicted_idx.item()]
    predicted_sci = scientific_names[predicted_class]
    conf_score = confidence.item() * 100

    with col2:
        st.markdown("### Hasil Analisis:")
        st.success(f"**Prediksi Kelas:** {predicted_class}")
        st.info(f"**Nama Ilmiah:** *{predicted_sci}*")
        st.metric(label="Tingkat Keyakinan (Confidence)", value=f"{conf_score:.2f}%")

    # Visualisasi Grad-CAM++ (XAI)
    st.markdown("---")
    st.subheader("Visualisasi Penjelasan Model (Explainable AI / Grad-CAM++)")
    
    try:
        # Normalisasi gambar untuk Grad-CAM [0-1]
        rgb_img = img_resized.astype(np.float32) / 255.0
        
        # Generate heatmap
        grayscale_cam = cam(input_tensor=tensor_img)[0]
        
        from pytorch_grad_cam.utils.image import show_cam_on_image
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        
        st.image(visualization, caption=f"Heatmap Grad-CAM++ pada kelas {predicted_class}", use_container_width=True)
    except Exception as e:
        st.warning(f"Gagal menghasilkan visualisasi Grad-CAM: {e}")

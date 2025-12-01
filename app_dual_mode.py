import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn

st.set_page_config(
    page_title="Dual-Mode AI Eye Screening",
    page_icon="🔬",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: bold;
        text-align: center;
        color: #1565C0;
        padding: 1.5rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .normal-result {
        background-color: #E8F5E9;
        border-left: 6px solid #4CAF50;
    }
    .abnormal-result {
        background-color: #FFEBEE;
        border-left: 6px solid #F44336;
    }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Mode 1: Binary (Normal vs Abnormal)
    model1 = models.resnet34()
    model1.fc = nn.Linear(model1.fc.in_features, 2)
    model1.load_state_dict(torch.load("models/best_model.pth", map_location=device))
    model1 = model1.to(device)
    model1.eval()
    
    # Mode 2: 4-class retinal
    model2 = models.resnet34()
    model2.fc = nn.Linear(model2.fc.in_features, 4)
    model2.load_state_dict(torch.load("models/retinal_model.pth", map_location=device))
    model2 = model2.to(device)
    model2.eval()
    
    return model1, model2, device

model1, model2, device = load_models()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Header
st.markdown('<p class="main-header">AI Eye Disease Screening System</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #424242;">Dual-Mode Classification System</p>', unsafe_allow_html=True)

# Sidebar - Mode Selection
with st.sidebar:
    st.header("Select Screening Mode")
    
    mode = st.radio(
        "Choose mode:",
        ["Mode 1: External Eye Disease", "Mode 2: Retinal Disease Screening"],
        help="Select the type of eye screening"
    )
    
    st.markdown("---")
    
    if "Mode 1" in mode:
        st.subheader("Mode 1 Details")
        st.write("**Type:** External Eye Classification")
        st.write("**Input:** Normal camera photos")
        st.write("**Accuracy:** 97.62%")
        st.write("**Classes:**")
        st.write("- Normal")
        st.write("- Abnormal")
    else:
        st.subheader("Mode 2 Details")
        st.write("**Type:** Retinal Disease Screening")
        st.write("**Input:** Fundus/retinal images")
        st.write("**Accuracy:** 91.51%")
        st.write("**Classes:**")
        st.write("- Normal")
        st.write("- Cataract")
        st.write("- Glaucoma")
        st.write("- Diabetic Retinopathy")
    
    st.markdown("---")
    st.info("**Disclaimer:** Screening tool only. Professional medical diagnosis required.")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    if "Mode 1" in mode:
        st.subheader("Upload or Capture External Eye Image")
        
        # Input method selection
        input_method = st.radio(
            "Select input method:",
            ["Upload Image", "Take Photo with Camera"],
            horizontal=True
        )
        
        if input_method == "Upload Image":
            st.write("Upload a clear photograph of an eye.")
            uploaded_file = st.file_uploader(
                "Select image file",
                type=['jpg', 'jpeg', 'png'],
                help="Supported formats: JPG, JPEG, PNG",
                key="upload_mode1"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')
                st.image(image, caption="Uploaded Image", use_container_width=True)
                
                if st.button("Analyze Image", type="primary", use_container_width=True):
                    with st.spinner("Processing..."):
                        img_tensor = transform(image).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            outputs = model1(img_tensor)
                            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                            predicted_class = torch.argmax(probabilities).item()
                            confidence = probabilities[predicted_class].item()
                        
                        class_names = ['Normal', 'Abnormal']
                        result = class_names[predicted_class]
                        
                        st.session_state['mode'] = 1
                        st.session_state['result'] = result
                        st.session_state['confidence'] = confidence
                        st.session_state['probs'] = probabilities.cpu().numpy()
                        st.rerun()
        
        else:  # Take Photo with Camera
            st.write("**Instructions:**")
            st.write("1. Allow camera access when prompted")
            st.write("2. Position your eye in the frame")
            st.write("3. Click the camera button to capture")
            
            camera_photo = st.camera_input("Capture eye image", key="camera_mode1")
            
            if camera_photo is not None:
                image = Image.open(camera_photo).convert('RGB')
                st.success("Photo captured successfully!")
                
                if st.button("Analyze Captured Image", type="primary", use_container_width=True):
                    with st.spinner("Processing..."):
                        img_tensor = transform(image).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            outputs = model1(img_tensor)
                            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                            predicted_class = torch.argmax(probabilities).item()
                            confidence = probabilities[predicted_class].item()
                        
                        class_names = ['Normal', 'Abnormal']
                        result = class_names[predicted_class]
                        
                        st.session_state['mode'] = 1
                        st.session_state['result'] = result
                        st.session_state['confidence'] = confidence
                        st.session_state['probs'] = probabilities.cpu().numpy()
                        st.rerun()
    
    else:  # Mode 2
        st.subheader("Upload Retinal Fundus Image")
        st.write("Upload a fundus/retinal image for disease screening.")
        
        uploaded_file = st.file_uploader(
            "Select image file",
            type=['jpg', 'jpeg', 'png'],
            help="Supported formats: JPG, JPEG, PNG",
            key="upload_mode2"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
            if st.button("Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    img_tensor = transform(image).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        outputs = model2(img_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                        predicted_class = torch.argmax(probabilities).item()
                        confidence = probabilities[predicted_class].item()
                    
                    class_names = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']
                    result = class_names[predicted_class]
                    
                    st.session_state['mode'] = 2
                    st.session_state['result'] = result
                    st.session_state['confidence'] = confidence
                    st.session_state['probs'] = probabilities.cpu().numpy()
                    st.session_state['class_names'] = class_names
                    st.rerun()

with col2:
    if 'result' in st.session_state:
        result = st.session_state['result']
        confidence = st.session_state['confidence']
        probs = st.session_state['probs']
        mode_num = st.session_state['mode']
        
        # Results box
        is_normal = result == "Normal"
        box_class = "normal-result" if is_normal else "abnormal-result"
        st.markdown(f'<div class="result-box {box_class}">', unsafe_allow_html=True)
        
        st.subheader("Analysis Results")
        
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Classification", result)
        with metric_col2:
            st.metric("Confidence", f"{confidence*100:.1f}%")
        
        st.markdown("---")
        
        if mode_num == 1:
            # Mode 1 recommendations
            if is_normal:
                st.markdown("**Assessment:**")
                st.success("No visible abnormalities detected.")
                st.markdown("**Recommendations:**")
                st.write("• Maintain regular eye care routine")
                st.write("• Annual comprehensive examination")
                st.write("• Monitor for vision changes")
            else:
                st.markdown("**Assessment:**")
                st.error("Potential abnormality detected.")
                st.markdown("**Actions:**")
                st.write("• Consult ophthalmologist within 7 days")
                st.write("• Document symptoms")
                st.write("• Avoid eye rubbing")
        else:
            # Mode 2 recommendations
            if is_normal:
                st.markdown("**Assessment:**")
                st.success("Retinal scan appears normal.")
                st.markdown("**Recommendations:**")
                st.write("• Continue annual retinal screenings")
                st.write("• Maintain healthy diet")
                st.write("• Control blood sugar if diabetic")
            else:
                st.markdown("**Assessment:**")
                st.error(f"Detected: {result}")
                st.markdown("**Urgent Actions:**")
                st.write("• Schedule ophthalmologist appointment immediately")
                st.write("• Bring this screening result")
                st.write("• Early treatment prevents vision loss")
                
                if result == "Cataract":
                    st.write("\n**About Cataract:**")
                    st.write("Clouding of eye lens. Treatable with surgery.")
                elif result == "Glaucoma":
                    st.write("\n**About Glaucoma:**")
                    st.write("Optic nerve damage. Requires immediate treatment.")
                elif result == "Diabetic Retinopathy":
                    st.write("\n**About Diabetic Retinopathy:**")
                    st.write("Diabetes-related retinal damage. Control blood sugar urgently.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Probabilities
        with st.expander("View Probability Distribution"):
            if mode_num == 1:
                st.write(f"**Normal:** {probs[0]*100:.2f}%")
                st.progress(float(probs[0]))
                st.write(f"**Abnormal:** {probs[1]*100:.2f}%")
                st.progress(float(probs[1]))
            else:
                class_names = st.session_state['class_names']
                for i, name in enumerate(class_names):
                    st.write(f"**{name}:** {probs[i]*100:.2f}%")
                    st.progress(float(probs[i]))

# Footer
st.markdown("---")
st.markdown("""
<p style="text-align: center; color: #757575; font-size: 0.9rem;">
    Dual-Mode AI Eye Screening | Mode 1: 97.62% | Mode 2: 91.51% | PyTorch + ResNet-34<br>
    For screening purposes only | Not a substitute for professional diagnosis
</p>
""", unsafe_allow_html=True)

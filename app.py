import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn

st.set_page_config(
    page_title="AI Eye Disease Screening",
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
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        font-size: 1.1rem;
        color: #424242;
        margin-bottom: 1rem;
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
    .metric-box {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = models.resnet34()
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Header
st.markdown('<p class="main-header">AI Eye Disease Screening System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Binary Classification: Normal vs Abnormal Eye Conditions</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 0.95rem; color: #757575;">Model Accuracy: 97.62% | Architecture: ResNet34</p>', unsafe_allow_html=True)

st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Model Information")
    st.write("")
    st.write("**Architecture:** ResNet34")
    st.write("**Dataset Size:** 839 images")
    st.write("**Test Accuracy:** 97.62%")
    st.write("**Training:** 30 epochs")
    st.write("")
    st.write("**Class Performance:**")
    st.write("- Normal: 97.5% accuracy")
    st.write("- Abnormal: 97.7% accuracy")
    st.markdown("---")
    st.write("**Model Metrics:**")
    st.write("- Precision: 97.6%")
    st.write("- Recall: 97.6%")
    st.write("- F1-Score: 97.6%")
    st.markdown("---")
    st.info("**Medical Disclaimer:** This screening tool is for preliminary assessment only. Professional medical diagnosis by a qualified ophthalmologist is required for definitive evaluation and treatment.")

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Eye Image")
    st.write("Please upload a clear, well-lit photograph of an eye for analysis.")
    
    uploaded_file = st.file_uploader(
        "Select image file",
        type=['jpg', 'jpeg', 'png'],
        help="Supported formats: JPG, JPEG, PNG"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Processing image..."):
                # Preprocess
                img_tensor = transform(image).unsqueeze(0).to(device)
                
                # Predict
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                    predicted_class = torch.argmax(probabilities).item()
                    confidence = probabilities[predicted_class].item()
                
                class_names = ['Normal', 'Abnormal']
                result = class_names[predicted_class]
                
                # Store in session state
                st.session_state['result'] = result
                st.session_state['confidence'] = confidence
                st.session_state['probs'] = probabilities.cpu().numpy()
                st.rerun()

with col2:
    if 'result' in st.session_state:
        result = st.session_state['result']
        confidence = st.session_state['confidence']
        probs = st.session_state['probs']
        
        # Results box
        box_class = "normal-result" if result == "Normal" else "abnormal-result"
        st.markdown(f'<div class="result-box {box_class}">', unsafe_allow_html=True)
        
        st.subheader("Analysis Results")
        
        # Metrics
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Classification", result)
        with metric_col2:
            st.metric("Confidence Level", f"{confidence*100:.1f}%")
        
        st.markdown("---")
        
        # Recommendation
        if result == "Normal":
            st.markdown("**Clinical Assessment:**")
            st.success("No visible abnormalities detected. Eyes appear healthy.")
            
            st.markdown("**Recommendations:**")
            st.write("• Maintain regular eye care routine")
            st.write("• Schedule annual comprehensive eye examination")
            st.write("• Continue healthy lifestyle practices")
            st.write("• Monitor for any changes in vision or eye appearance")
            st.write("• Consult ophthalmologist if symptoms develop")
            
        else:
            st.markdown("**Clinical Assessment:**")
            st.error("Potential abnormality detected. Medical evaluation recommended.")
            
            st.markdown("**Recommended Actions:**")
            st.write("• Schedule appointment with ophthalmologist within 7 days")
            st.write("• Bring this screening result to your consultation")
            st.write("• Document any symptoms (redness, pain, discharge, vision changes)")
            st.write("• Avoid touching or rubbing eyes")
            st.write("• Do not delay professional evaluation")
            
            st.markdown("**Potential Conditions:**")
            st.write("This screening cannot specify the exact condition. Possible causes include:")
            st.write("- Conjunctivitis (bacterial or viral)")
            st.write("- Blepharitis (eyelid inflammation)")
            st.write("- Allergic conjunctivitis")
            st.write("- Stye or chalazion")
            st.write("- Other ocular surface disorders")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Probabilities
        with st.expander("View Detailed Probability Distribution"):
            st.write(f"**Normal Classification:** {probs[0]*100:.2f}%")
            st.progress(float(probs[0]))
            st.write("")
            st.write(f"**Abnormal Classification:** {probs[1]*100:.2f}%")
            st.progress(float(probs[1]))

# Footer
st.markdown("---")
st.markdown("""
<p style="text-align: center; color: #757575; font-size: 0.9rem;">
    AI Eye Disease Screening System | PyTorch + ResNet34 | 839 Training Images | 97.62% Test Accuracy<br>
    For educational and screening purposes only | Not a substitute for professional medical diagnosis
</p>
""", unsafe_allow_html=True)

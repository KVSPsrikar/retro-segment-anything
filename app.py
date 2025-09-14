import streamlit as st
import numpy as np
import cv2
from PIL import Image
import torch
import os
import io
from segment_anything import sam_model_registry, SamPredictor
from streamlit.components.v1 import html as components_html

# Load SAM model
@st.cache_resource
def load_model():
    try:
        sam_checkpoint = r"C:\Users\HP\checkpoints\sam_vit_b_01ec64.pth"
        model_type = "vit_b"
        
        if not os.path.exists(sam_checkpoint):
            st.error(f"Model file not found at: {sam_checkpoint}")
            st.stop()
            
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        predictor = SamPredictor(sam)
        return predictor
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

# Initialize the model
predictor = load_model()

# Set page config
st.set_page_config(
    page_title="Segment Anything",
    page_icon="🎯",
    layout="wide"
)

# Initialize session state
if 'image' not in st.session_state:
    st.session_state.image = None
if 'mask' not in st.session_state:
    st.session_state.mask = None
if 'points' not in st.session_state:
    st.session_state.points = []
if 'labels' not in st.session_state:
    st.session_state.labels = []
if 'click_data' not in st.session_state:
    st.session_state.click_data = None
if 'click_handler_key' not in st.session_state:
    st.session_state.click_handler_key = 0

# Sidebar
st.sidebar.title("Segment Anything")
st.sidebar.markdown("### Model Settings")
model_type = st.sidebar.selectbox(
    "Model Type",
    ["Base (vit_b)", "Large (vit_l)", "Huge (vit_h)"],
    index=0
)

# Add sample images
st.sidebar.markdown("### Sample Images")
sample_images = {
    "Dog": "notebooks/images/dog.jpg",
    "Truck": "notebooks/images/truck.jpg",
    "Groceries": "notebooks/images/groceries.jpg"
}

selected_sample = st.sidebar.selectbox("Load Sample", list(sample_images.keys()))
if st.sidebar.button("Load Selected Sample"):
    try:
        image_path = sample_images[selected_sample]
        if not os.path.exists(image_path):
            st.sidebar.error(f"Sample image not found at: {image_path}")
        else:
            st.session_state.image = cv2.cvtColor(
                cv2.imread(image_path),
                cv2.COLOR_BGR2RGB
            )
            st.session_state.mask = None
            st.session_state.points = []
            st.session_state.labels = []
            st.session_state.click_data = None
            st.rerun()
    except Exception as e:
        st.sidebar.error(f"Error loading sample image: {str(e)}")

# File uploader
uploaded_file = st.sidebar.file_uploader("Or upload your own image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    st.session_state.image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.session_state.image = cv2.cvtColor(st.session_state.image, cv2.COLOR_BGR2RGB)
    st.session_state.mask = None
    st.session_state.points = []
    st.session_state.labels = []
    st.session_state.click_data = None

# Main content
st.title("Segment Anything")
st.markdown("Click on the image to add foreground points (left-click) or background points (right-click)")

if st.session_state.image is not None:
    # Display the image
    st.image(st.session_state.image, width='stretch', caption="Click to add points")
    
    # Handle click data from JavaScript
    click_data = components_html(
        """
        <script>
        function handleClick(e) {
            const rect = this.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            const label = e.which === 1 ? 1 : 0;
            
            const data = {
                x: x / rect.width,
                y: y / rect.height,
                label: label
            };
            
            if (window.parent.Streamlit) {
                window.parent.Streamlit.setComponentValue(data);
            }
        }

        function setupClickHandler() {
            const images = document.querySelectorAll('.stImage img');
            if (images.length > 0) {
                const img = images[0];
                img.style.cursor = 'crosshair';
                img.removeEventListener('click', handleClick);
                img.removeEventListener('contextmenu', handleContextMenu);
                img.addEventListener('click', handleClick);
                img.addEventListener('contextmenu', handleContextMenu);
            }
        }

        function handleContextMenu(e) {
            e.preventDefault();
            handleClick.call(this, e);
            return false;
        }

        // Initial setup
        document.addEventListener('DOMContentLoaded', setupClickHandler);
        const observer = new MutationObserver(setupClickHandler);
        observer.observe(document.body, { childList: true, subtree: true });
        </script>
        """,
        height=0
    )

    # Update the key to force re-render
    st.session_state.click_handler_key += 1

    # Handle click data
    if click_data is not None:
        st.session_state.click_data = click_data
        st.rerun()

    if 'click_data' in st.session_state and st.session_state.click_data:
        data = st.session_state.click_data
        if st.session_state.image is not None:
            h, w = st.session_state.image.shape[:2]
            x = int(data['x'] * w)
            y = int(data['y'] * h)
            st.session_state.points.append([x, y])
            st.session_state.labels.append(data['label'])
            st.session_state.click_data = None
            st.rerun()

    # Add point collection
    points = st.session_state.get('points', [])
    
    # Add click handler
    if st.button("Clear Points"):
        st.session_state.points = []
        st.session_state.labels = []
        st.session_state.mask = None
        st.session_state.click_data = None
        st.rerun()
    
    # Handle clicks on the image
    if st.session_state.points:
        # Convert points to numpy arrays
        input_points = np.array(st.session_state.points)
        input_labels = np.array(st.session_state.labels)
        
        try:
            # Run SAM
            predictor.set_image(st.session_state.image)
            masks, scores, _ = predictor.predict(
                point_coords=input_points,
                point_labels=input_labels,
                multimask_output=True,
            )
            
            # Get the best mask
            if len(masks) > 0:
                st.session_state.mask = masks[0]
                # Display the mask
                st.image(st.session_state.mask, width='stretch', caption="Segmentation Mask")
        except Exception as e:
            st.error(f"Error during segmentation: {str(e)}")
    
    # Add download button if mask exists
    if st.session_state.mask is not None:
        # Convert mask to image
        mask_image = Image.fromarray((st.session_state.mask * 255).astype(np.uint8))
        # Save to bytes
        img_byte_arr = io.BytesIO()
        mask_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        # Create download button
        st.download_button(
            label="Download Mask",
            data=img_byte_arr,
            file_name="mask.png",
            mime="image/png"
        )

# Add footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9em; margin-top: 30px;">
        <p>© 2025 Segment Anything Web App | Built with Meta's Segment Anything Model</p>
    </div>
""", unsafe_allow_html=True)
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import io

class FaceAnonymizer:
    def __init__(self):
        # loads harcascade for facial detecition
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
      
    def detect_faces(self, image):
        """ 
        input : takes an image 
        output : returns list of rectangles, each rectangle represent a face 
        [[(100, 50, 80, 80), (250, 60, 85, 85)] : means two faces were detected. 
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    
    def pixelate_area(self, image, x, y, w, h, pixel_size=15):
        """
        input : image,
                (x,y) is the top-left corner of the rectangle
                (w,h) is the width and height of the rectangle
        output : returns the image with the selected area pixelated.
        """
        reason_of_interest = image[y:y+h, x:x+w]
        downscaled_roi = cv2.resize(reason_of_interest, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(downscaled_roi, (w, h), interpolation=cv2.INTER_NEAREST)
        image[y:y+h, x:x+w] = pixelated
        return image
    
    def blur_area(self, image, x, y, w, h, blur_strength=25):
        """Apply gaussian blur to a specific area"""
        roi = image[y:y+h, x:x+w]
        # Ensure blur strength is odd
        if blur_strength % 2 == 0:
            blur_strength += 1
        blurred = cv2.GaussianBlur(roi, (blur_strength, blur_strength), 0)
        image[y:y+h, x:x+w] = blurred
        return image
    
    def process_image(self, image, method='blur', pixel_size=15, blur_strength=25, padding=10):
        """Process an image to anonymize faces"""
        result = image.copy()
        faces = self.detect_faces(image)
        
        for (x, y, w, h) in faces:
            # Add padding around the face
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            if method == 'pixelate':
                result = self.pixelate_area(result, x, y, w, h, pixel_size)
            elif method == 'blur':
                result = self.blur_area(result, x, y, w, h, blur_strength)
                
        return result, len(faces)

# helper functions to convert PIL to CV2
def pil_to_cv2(pil_image):
    open_cv_image = np.array(pil_image.convert('RGB'))
    return cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)

# helper functions to convert CV2 to PIL
def cv2_to_pil(cv2_image):
    """Convert OpenCV image to PIL format"""
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)

def main():
    st.set_page_config(
        page_title="Face Anonymizer",
        page_icon="🙈",
        layout="wide"
    )
    
    st.title("Face Anonymizer")
    st.markdown("Upload an image and automatically blur or pixelate faces for privacy protection")
    

    if 'anonymizer' not in st.session_state:
        st.session_state.anonymizer = FaceAnonymizer()
    
    st.sidebar.header("Settings")
    
    method = st.sidebar.selectbox(
        "Anonymization Method",
        ["blur", "pixelate"],
        help="Choose between blur or pixelation effect"
    )
    
    if method == "blur":
        blur_strength = st.sidebar.slider(
            "Blur Strength",
            min_value=5,
            max_value=99,
            value=25,
            step=2,
            help="Higher values = more blur (must be odd)"
        )
       
        if blur_strength % 2 == 0:
            blur_strength += 1
    else:
        pixel_size = st.sidebar.slider(
            "Pixel Size",
            min_value=5,
            max_value=50,
            value=15,
            help="Lower values = more pixelated"
        )
    
    padding = st.sidebar.slider(
        "Face Padding",
        min_value=0,
        max_value=50,
        value=10,
        help="Adds an extra padding around detected faces"
    )
    
    # upload a file 
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a JPG, PNG or image"
    )
    
    if uploaded_file is not None:
        # if image is uploaded open and display the image 
        pil_image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📸 Original Image")
            st.image(pil_image, use_column_width=True)
        
        # process the image
        with st.spinner("detecting and anonymizing faces"):
            # convert PIL to cv2 format 
            cv2_image = pil_to_cv2(pil_image)
            
            # process based on selected method
            if method == "blur":
                processed_image, face_count = st.session_state.anonymizer.process_image(
                    cv2_image, method=method, blur_strength=blur_strength, padding=padding
                )
            else:
                processed_image, face_count = st.session_state.anonymizer.process_image(
                    cv2_image, method=method, pixel_size=pixel_size, padding=padding
                )
            
            # convert back to PIL for display
            result_pil = cv2_to_pil(processed_image)
        
        with col2:
            st.subheader("Anonymized Image")
            st.image(result_pil, use_column_width=True)
        
        # Show results info
        if face_count > 0:
            st.success(f"Successfully anonymized {face_count} face(s) using {method}")
        else:
            st.warning("No faces detected in the image")
        
       
        img_buffer = io.BytesIO()
        result_pil.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        
        st.download_button(
            label="Download Anonymized Image",
            data=img_buffer.getvalue(),
            file_name=f"anonymized_{uploaded_file.name}",
            mime="image/png",
            use_container_width=True
        )
        
        # Settings info
        with st.expander("ℹ️ Processing Details"):
            st.write(f"**Method:** {method.title()}")
            if method == "blur":
                st.write(f"**Blur Strength:** {blur_strength}")
            else:
                st.write(f"**Pixel Size:** {pixel_size}")
            st.write(f"**Face Padding:** {padding}px")
           

if __name__ == "__main__":
    main()
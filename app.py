import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from PIL import Image

# Load your trained model
model = load_model('braintumor.h5')  # Update the path if necessary

# Define your Streamlit app
st.markdown("<h1 style='text-align: center; color: #00e3f8;'>Brain Tumor Classification</h1>", unsafe_allow_html=True)

# Add a sidebar with logo, contact info, and description
st.sidebar.image('H.png', width=250)
st.sidebar.info('This app classifies brain tumor images into four categories.')
st.sidebar.markdown("[Learn More](https://github.com/HajarHal/brain_tumour)")


# Custom CSS for sidebar
# Apply background color to the sidebar
st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        background-color: #0074D9;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Function to make predictions
def predict(image):
    # Preprocess the image if necessary
    # For example, resize it to match the input size of your model
    image = cv2.resize(image, (150, 150))
    image = image.reshape(1, 150, 150, 3)  # Reshape for model input
    # Make prediction
    prediction = model.predict(image)
    return prediction

# Main app code
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make prediction on the uploaded image
    prediction = predict(image)
    labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    predicted_label = labels[np.argmax(prediction)]

    # Display the result based on the predicted label
    if predicted_label == 'no_tumor':
        st.write('**Prediction:** ', f'<span style="color:green; font-weight:bold">Normal</span>', unsafe_allow_html=True)
    elif predicted_label == 'meningioma_tumor':
        st.write('**Prediction:** ', f'<span style="color:red; font-weight:bold">Meningioma Tumor</span>', unsafe_allow_html=True)
    elif predicted_label == 'glioma_tumor':
        st.write('**Prediction:** ', f'<span style="color:red; font-weight:bold">Glioma Tumor</span>', unsafe_allow_html=True)
    elif predicted_label == 'pituitary_tumor':
        st.write('**Prediction:** ', f'<span style="color:red; font-weight:bold">Pituitary Tumor</span>', unsafe_allow_html=True)

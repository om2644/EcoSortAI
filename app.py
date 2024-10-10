import tensorflow as tf
import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from PIL import Image

# Load the trained model
model = load_model('model/Image_classify.keras')

# Define categories
data_cat = ['Organic', 'Recyclable']

# Define image dimensions based on the model's expected input shape
img_height = 180  # Height defined in the model
img_width = 180   # Width defined in the model

# Set header and description
st.title("EcoSort AI: Waste Classification")
st.write("Point your camera at waste, and this AI will classify it as either Organic or Recyclable!")

# Function to make predictions on an image
def predict_frame(image):
    # Resize image to model's expected input shape
    image = image.resize((img_width, img_height))
    img_arr = tf.keras.utils.img_to_array(image)
    img_bat = tf.expand_dims(img_arr, axis=0)

    # Make predictions
    predict = model.predict(img_bat)
    score = tf.nn.softmax(predict)

    # Get prediction and confidence
    category = data_cat[np.argmax(score)]
    confidence = np.max(score) * 100  # Convert to percentage
    return category, confidence

# Create a class to process the video frames from the webcam
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.img_height = img_height
        self.img_width = img_width

    def transform(self, frame):
        # Convert frame to numpy array (BGR format)
        img = frame.to_ndarray(format="bgr")
        
        # Resize frame to the model's input size
        img_resized = cv2.resize(img, (self.img_width, self.img_height))
        img_arr = tf.keras.utils.img_to_array(img_resized)
        img_bat = tf.expand_dims(img_arr, axis=0)

        # Make predictions using the model
        predict = model.predict(img_bat)
        score = tf.nn.softmax(predict)

        # Get the prediction and confidence
        category = data_cat[np.argmax(score)]
        confidence = np.max(score) * 100  # Convert to percentage

        # Display the prediction on the frame
        label = f"{category} ({confidence:.2f}%)"
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

# Use WebRTC to access the webcam through the browser
webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

# Upload image functionality for classification
uploaded_file = st.file_uploader("Upload an image...", type=["jpeg", "jpg", "png"])

if uploaded_file is not None:
    # Load and display the uploaded image
    uploaded_image = Image.open(uploaded_file)

    # Create columns for layout
    col1, col2 = st.columns(2)

    # Display uploaded image in the first column
    with col1:
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    # Make predictions on the uploaded image and display in the second column
    with col2:
        category, confidence = predict_frame(uploaded_image)
        st.markdown(f"<h2 style='text-align: center;'><strong>Prediction: {category}<strong></h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center;'>Confidence: <b>{confidence:.2f}%</b></p>", unsafe_allow_html=True)

# Instruction for using the app
st.info("Start your webcam or upload an image to get a waste classification prediction.")

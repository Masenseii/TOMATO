# Importing necessary libraries
import os
import cv2
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import csv
import io

# For image capture
import time
import threading
import subprocess

# Class names
class_name = ['Bacterial spot__Blight',
              'Not Recognized',
              'Tomato Healthy',
              'Tomato Leaf Miner Flies',
              'Tobacco Caterpillar',
              'Tomato Leaf Curl']

# Loading the trained model
@st.cache_resource
def load_model():
  try:
    model = tf.keras.models.load_model('Tomato_model_best.keras')
    st.write("Model loaded")
    print(model.summary())
    return model
  except Exception as e:
    st.error(f"Error loading the model: {e}")
    return None

# Function to capture images by the CSI Camera
def capture_image(folder="Captured_Images"):
  if not os.path.exists(folder):
    os.mkdirs(folder)
    image_path = os.path.join(folder, "captured_image.jpg")
    subprocess.run("[libcamera-still", "-o", image_path])
    return image_path

# Initializing session state for the captured image
if "captured_image" not in st.session_state:
  st.session_state.captured_image = None

# Define function for prediction
# Preprocess the image using keras.preprocessing

  

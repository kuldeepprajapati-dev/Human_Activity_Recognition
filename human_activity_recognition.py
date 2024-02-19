import streamlit as st
import cv2
import tensorflow as tf 
from tensorflow.keras.models import load_model
import numpy as np
import os

model = load_model('Save_Model.h5')
classes = ['Javelin Throw', 'High Jump', 'Biking', 'Playing Piano', 'Basketball']

st.title('Human Activity Recognition')
video = st.file_uploader('Upload your video', type=['mp4', 'mov', 'avi', 'mkv'])

if video is not None:
    with open("temp_video.mp4", "wb") as f:
        f.write(video.read())
    st.video("temp_video.mp4")

def predict(classes):
    reader = cv2.VideoCapture("temp_video.mp4")
    frames_length = 20
    image_height = 64
    image_width = 64
    test_frames = []
    total_frames_count = reader.get(cv2.CAP_PROP_FRAME_COUNT)
    skip_window = int(total_frames_count / frames_length)
    for k in np.arange(frames_length):
        reader.set(cv2.CAP_PROP_POS_FRAMES,(k*skip_window))
        _, frame =  reader.read()
        frame_resize = cv2.resize(frame , (image_height, image_width))
        frame_normalized =  (frame_resize)/255.0
        test_frames.append(frame_normalized)
    reader.release()
    test_frames =  np.array(test_frames)
    test_frames =  np.expand_dims(test_frames, axis = 0)
    # print(np.argmax(model.predict(test_frames)))
    st.info(f"Predicted Label : {classes[np.argmax(model.predict(test_frames))]}")

if st.button('Predict'):
    try:
        predict(classes)
    except Exception as e:
        st.write('Error:', e)

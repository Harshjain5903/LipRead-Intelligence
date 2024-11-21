# Import all of the dependencies
import streamlit as st
import os
import imageio
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model

# Set the layout to the Streamlit app as wide
st.set_page_config(layout='wide')

# Setup the sidebar
with st.sidebar:
    st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
    st.title('LipBuddy')
    st.info('This application is originally developed from the LipNet deep learning model.')

st.title('LipNet Full Stack App')

# Define paths to video files and alignment data
video_path = '/Users/harshjain/Downloads/LipNet-main/data/s1'

# Generating a list of video options
try:
    options = os.listdir(video_path)
    if len(options) == 0:
        st.error("No video files found in the specified directory.")
    else:
        selected_video = st.selectbox('Choose a video', options)
except FileNotFoundError:
    st.error(f"Directory not found: {video_path}")
    st.stop()

# Generate two columns for layout
col1, col2 = st.columns(2)

if options:
    # Rendering the video in Column 1
    with col1:
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join(video_path, selected_video)

        # Use ffmpeg to convert the selected video to mp4 format
        converted_video_path = os.path.join('/Users/harshjain/Downloads/LipNet-main/app', 'converted_video.mp4')
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 {converted_video_path} -y')

        # Render the converted video inside the app
        try:
            video = open(converted_video_path, 'rb')
            video_bytes = video.read()
            st.video(video_bytes)
        except FileNotFoundError:
            st.error("Converted video file not found. Please check the conversion process.")

    # Processing and displaying ML predictions in Column 2
    with col2:
        st.info('This is all the machine learning model sees when making a prediction')

        # Load data and annotations for the selected video
        try:
            video_tensor, annotations = load_data(tf.convert_to_tensor(file_path))
            imageio.mimsave('animation.gif', video_tensor, fps=10)
            st.image('animation.gif', width=400)
        except Exception as e:
            st.error(f"Failed to load video data: {e}")

        st.info('This is the output of the machine learning model as tokens')

        # Load the pre-trained model and make predictions
        try:
            model = load_model()
            yhat = model.predict(tf.expand_dims(video_tensor, axis=0))
            decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
            st.text(decoder)
        except Exception as e:
            st.error(f"Model prediction failed: {e}")
            decoder = None  # Initialize decoder to avoid undefined error later

        # Convert prediction to text
        if decoder is not None:
            try:
                converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
                st.text(converted_prediction)
            except Exception as e:
                st.error(f"Decoding failed: {e}")
        else:
            st.error("No valid decoder output available for text conversion.")

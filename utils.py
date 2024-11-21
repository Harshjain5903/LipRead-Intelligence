import tensorflow as tf
from typing import List
import cv2
import os

# Define vocabulary for character-to-number mapping
vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
# Mapping integers back to original characters
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)


def load_video(path: str) -> List[float]:
    """
    Load video frames and normalize them.

    Args:
        path (str): The path to the video file.

    Returns:
        List[float]: A list of processed frames.
    """
    # Capture video using OpenCV
    cap = cv2.VideoCapture(path)
    frames = []

    # Iterate through each frame in the video
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT()))):
        ret, frame = cap.read()
        if not ret:
            break
        # Convert frame to grayscale
        frame = tf.image.rgb_to_grayscale(frame)
        # Crop and append frame
        frames.append(frame[190:236, 80:220, :])

    cap.release()

    # Normalize frames
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std


def load_alignments(path: str) -> List[str]:
    """
    Load alignment tokens from the alignment file.

    Args:
        path (str): The path to the alignment file.

    Returns:
        List[str]: A list of character tokens.
    """
    # Read alignment file
    with open(path, 'r') as f:
        lines = f.readlines()

    tokens = []
    # Extract tokens from each line
    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            tokens = [*tokens, ' ', line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]


def load_data(path: str):
    """
    Load video data and alignments for a given file path.

    Args:
        path (str): The path to the video file.

    Returns:
        Tuple: A tuple containing video frames and alignment tokens.
    """
    # Decode the path
    path = bytes.decode(path.numpy())

    # Extract file name
    file_name = path.split('/')[-1].split('.')[0]

    # File name splitting for windows
    # Uncomment the below line if you're using Windows and comment the above line
    # file_name = path.split('\\')[-1].split('.')[0]

    # Construct paths for video and alignment files
    video_path = os.path.join('/Users/harshjain/Downloads/LipNet-main/data/s1', f'{file_name}.mp4')
    alignment_path = os.path.join('/Users/harshjain/Downloads/LipNet-main/data/alignments/s1', f'{file_name}.align')

    # Load frames and alignments
    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)

    return frames, alignments

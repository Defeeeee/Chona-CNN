import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

# Load your model
model = tf.keras.models.load_model('notmnist_model1.h5')

st.title('My 28x28 Image Model')

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    img = image.resize((28, 28))
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for the model (add batch dimension)

    # Make prediction
    prediction = model.predict(img_array)

    # Display the prediction
    st.write(f"Prediction: {prediction}")

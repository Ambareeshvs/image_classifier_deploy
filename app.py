import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
from tensorflow.keras import datasets, layers, models
from keras.preprocessing import image


st.title("Image Classification")
st.header("Ship Truck Classification")
st.text("Upload an image..")

@st.cache(allow_output_mutation=True)
def load_model():
  model=tf.keras.models.load_model('truck-ship-classifier.hdf5') # Loading the cnn model that is created
  return model

model=load_model()

file = st.file_uploader("Please upload a file", type=["jpg", "png","jpeg"])

st.set_option('deprecation.showfileUploaderEncoding', False)
def loading_and_predicting(pic, model):
        size = (32,32)    
        image = ImageOps.fit(pic, size, Image.ANTIALIAS)       # Adjusting size of img as in model
        image = np.asarray(image)                              # Converting to array
        img = image.astype(np.float32) / 255.0                 # Normalising as in the model
        final_img = img[np.newaxis,...]                        # Adding extra dimension since training uses large data
        prediction = model.predict(final_img)
        return prediction


if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)                     # Displaying the uploaded image
    predictions = loading_and_predicting(image, model)
    value = np.argmax(predictions)                             
    if value == 9:
        st.markdown("It seems like a Truck")
    elif value == 8:
        st.markdown("It seems like a Ship")
    else:
        st.markdown("Sorry..! Can't classify")
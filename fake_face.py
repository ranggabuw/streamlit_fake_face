import streamlit as st
import tensorflow as tf
import numpy as np
import random
from PIL import Image, ImageOps
from pyngrok import ngrok
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Fake Face Recognation",
    page_icon = ":smile:",
    initial_sidebar_state = 'auto'
)

with st.sidebar:
        st.image('dekrip.png')
        st.title("Face")
        st.subheader("Accurate detection of the face. This helps an user to easily detect who has a fake face in social media.")

st.write("""
         # Fake Face Recognation
         """
         )

file = st.file_uploader("", type=["jpg", "png"])

def import_img(image_data):
    size = (224,224)    
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,:,:]
    return img_reshape

def predict(image_data, model):
    prediction = model.predict(import_img(image_data))
    return prediction

# def import_and_predict(image_data, model):
#         size = (224,224)    
#         image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
#         img = np.asarray(image)
#         img_reshape = img[np.newaxis]
#         prediction = model.predict(img_reshape)
#         # prediction = np.argmax(model.predict([img], verbose=0))
#         # if prediction == 0:
#         #     return 'Fake Face'
#         # else:
#         #     return 'Real Face'
#         return prediction

# st.set_option('deprecation.showfileUploaderEncoding', False)
# @st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('fake_face.h5')
    return model

model = load_model()

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    img = import_img(image)
    prediction = predict(image, model)
    # prediction = import_and_predict(image, model)
    class_index = np.argmax(model.predict(img))
    st.sidebar.success(f"Accuracy : {np.round(model.predict(img, verbose=0)[0][class_index]*100, 2)} %")

    class_names = ["Fake Face", "Real Face"]
    string = "The Face is " + class_names[np.argmax(prediction)]

    if class_names[np.argmax(prediction)] == "Real Face":
        st.balloons()
        st.sidebar.success(string)
    else:
        st.sidebar.error(string)
        st.info("### The Face is Fake ☹️")

import streamlit as st 
from PIL import Image
import os




st.title("JapanSelfieGAN")
st.header("From Selfie to Japan drawing style Portrait with GAN Model") 
st.markdown("")



#path 
"""
path = os.path.join("selfie","JosuéLouis-Alexandre_INVC21HS-min.jpg")
path2 = os.path.join("selfie","AhmedDeyab_INVC21HS.jpg")
path3 = os.path.join("selfie","Anais_Cisneros_INVC21HS.png")
path4 = os.path.join("selfie","IMG_20190920_152402_706.jpg")
path5 = os.path.join("selfie","ShaunaJin_INVC21HS.jpg")

#showing img 
#img = Image.open(path)
#img = img.resize((400,400))
st.image(img)
"""
#file uploader
st.header('Please provide a selfie to transform')
upload_files = st.file_uploader('', accept_multiple_files=False)
#showing uploaded img 
if upload_files is not None: 
    uploaded_img = Image.open(upload_files)
    uploaded_img = uploaded_img.resize((400,400))
    st.image(uploaded_img)




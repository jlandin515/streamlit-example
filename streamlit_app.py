from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
from PIL import Image


def load_image(image_file):
	img = Image.open(image_file)
	return img

def main():
  st.title("Aplicación de Clasificación de Lenguaje de Señas")

  menu = ["Image","Dataset","DocumentFiles","About"]
  choice = st.sidebar.selectbox("Menu",menu)

  if (choice == "Image"):
    st.subheader("Image")
  elif (choice == "Dataset"):
    st.subheader("Dataset")
  elif (choice == "DocumentFiles"):
    st.subheader("DocumentFiles")

  if choice == "Image":
    st.subheader("Image")
    image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

    if image_file is not None:
      file_details = {"filename":image_file.name, "filetype":image_file.type,
                              "filesize":image_file.size}
      st.write(file_details)
      st.image(load_image(image_file),width=250)

main()
from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
from PIL import Image

'''
# Aplicación

'''

def load_image(image_file):
	img = Image.open(image_file)
	return img

def main():
  st.title("File Upload Tutorial")

  menu = ["Image","Dataset","DocumentFiles","About"]
  choice = st.sidebar.selectbox("Menu",menu)

  if (choice == "Image"):
    st.subheader("Image")
  elif (choice == "Dataset"):
    st.subheader("Dataset")
  elif (choice == "DocumentFiles"):
    st.subheader("DocumentFiles")

main()
from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

"""
# Aplicación de Clasificación de Imágenes de Lenguaje de Señas 

"""

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
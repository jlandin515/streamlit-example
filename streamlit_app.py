from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
from PIL import Image
import torch
import torch.optim as optim
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

def load_image(image_file):
	img = Image.open(image_file)
	return img

def load_model():
  model = model.load_state_dict(torch.load('model_pro1.pt'))
  return model

def main():
  st.title("Aplicación de Clasificación de Lenguaje de Señas")

  menu = ["Clasificar Imagen","Proceso de Modelamiento","Evaluación de Testing"]
  choice = st.sidebar.selectbox("Menu",menu)

  if (choice == "Clasificar Imagen"):
    st.subheader("Clasificar Imagen")
  elif (choice == "Proceso de Modelamiento"):
    st.subheader("Proceso de Modelamiento")
  elif (choice == "Evaluación de Testing"):
    st.subheader("Evaluación de Testing")

  if choice == "Clasificar Imagen":
    image_file = st.file_uploader("Subir Imagen", type=["png","jpg","jpeg"])

    if image_file is not None:
      file_details = {"filename":image_file.name, "filetype":image_file.type,
                              "filesize":image_file.size}
      #st.write(file_details)
      st.image(load_image(image_file),width=250)

  model = load_model()
  output = model(image_file)
  st.success(output)
main()
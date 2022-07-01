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
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(128 * 28 * 28, 600)
            self.fc2 = nn.Linear(600, 27)
            self.dropout = nn.Dropout(0.2)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 128 * 28 * 28)
            x = self.dropout(x)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    model = Net()
    model.load_state_dict(torch.load('model_pro1.pt'))
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
      classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','NAN','O','P','Q','R','S','T','U','V','W','X','Y','Z']
      img = load_image(image_file)
      to_tensor = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224), transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
      tensor = to_tensor(img)
      tensor = tensor.unsqueeze(0)

      model.cpu()
      model.eval()
  
      output = model(tensor)
      _, preds_tensor = torch.max(output, 1)
      preds = np.squeeze(preds_tensor.cpu().numpy())
      st.success("La letra es " + classes[preds])


if __name__ == '__main__':
	main()
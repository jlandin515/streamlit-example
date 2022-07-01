from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st

"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:

If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

def main():
	st.title("File Upload Tutorial")

	menu = ["Image","Dataset","DocumentFiles","About"]
	choice = st.sidebar.selectbox("Menu",menu)

	if choice == "Image":
		st.subheader("Image")

    elif choice == "Dataset":
		st.subheader("Dataset")

    elif choice == "DocumentFiles":
		st.subheader("DocumentFiles")
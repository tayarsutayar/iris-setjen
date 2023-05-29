import streamlit as st
import pandas as pd
import streamlit.components.v1 as stc 
import joblib
from PIL import Image
from prediction import run_predict_app
from eda import run_eda_app

st.set_page_config(page_title="Iris",
		   layout="wide")

html_temp = """
		<div style="background-color:#3872fb;padding:5px;border-radius:10px">
		<h3 style="color:white;text-align:center;font-family:arial;">Iris </h3>
		<h3 style="color:white;text-align:center;font-family:arial;">PJJ DAS II 2023</h3>
		</div>
		"""


def main():
    stc.html(html_temp)
    menu = ["Home","EDA","Prediction"]
    choice = st.sidebar.selectbox("Menu",menu)
    if choice == "Home":
        st.subheader('Prediksi Iris - Sekretariat Jenderal')
        st.image(Image.open('iris.png'))
        st.write("""
                ###### APP Content:
                EDA Section: Exploratory Data Analysis of Data | Prediction Section: ML Predictor App
			""")
    elif choice == "EDA":
        run_eda_app()

    elif choice == "Prediction" :
        run_predict_app()



if __name__=='__main__':
    main()
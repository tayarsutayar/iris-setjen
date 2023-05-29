import streamlit as st
import joblib
import pandas as pd
import numpy as np
from PIL import Image

def load_model():
	model_gb=joblib.load('model.joblib')
	return model_gb

def run_predict_app():
    st.subheader("Prediction Section")
    model=load_model()
    with st.sidebar:
        st.title("Features")
        sepal_length = st.slider('Sepal Length',0, 10, value=4)
        sepal_width = st.slider('Sepal Width',0, 10, value=4)
        petal_length = st.slider('Petal Length',0, 10, value=4)
        petal_width = st.slider('Petal Width',0, 10, value=4)
        
     
    if st.button("Click here to predict"):
        st.info('Input :')
        st.write('Sepal Length : {} '.format(sepal_length))
        st.write('Sepal Width : {} '.format(sepal_width))
        st.write('Petal Length : {} '.format(petal_length))
        st.write('Petal Width : {} '.format(petal_width))
        dfvalues = pd.DataFrame(list(zip([sepal_length],[sepal_width],[petal_length],[petal_width])),columns =['sepal_length','sepal_width','petal_length','petal_width'])
       
        input_variables=np.array(dfvalues)

        prediction = model.predict(input_variables)
        pred_prob = model.predict_proba(input_variables)
        st.info('Result :')
        col1,col2 = st.columns([1,2])
        with col1:
            st.write('Prediction :', prediction[0])
            if prediction[0] == 'Iris-setosa':
                st.image(Image.open('setosa.png'), width=460)
            elif prediction[0] == 'Iris-versicolor':
                st.image(Image.open('versicolor.png'), width=460)
            elif prediction[0] == 'Iris-virginica':
                st.image(Image.open('virginica.png'), width=460)
        
        with col2:
            pred_probability_score = pred_prob[0][0]*100
            st.write("Prediction Probability Score : {:.2f} %".format(pred_probability_score))
              
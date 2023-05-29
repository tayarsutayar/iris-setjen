import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


def load_data(data):
    return pd.read_csv(data)

def rename_columns(dataframe):
    old_names = ['SepalLengthCm', 'SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']
    new_names = ['sepal_length', 'sepal_width', 'petal_length','petal_width','species']
    new_columns = dict(zip(old_names, new_names))
    return dataframe.rename(columns=new_columns)

def normalize(data):
	scaler = MinMaxScaler()
	ndf = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
	return ndf


def run_eda_app():
    st.subheader('EDA Section')

    df = load_data('iris_dataset.csv')

    df = rename_columns(df)
    
    submenu = st.sidebar.selectbox("Submenu",['Descriptive','Statistik'])
    if submenu=='Descriptive':
        st.write('Descriptive')
        with st.expander('Data Frame'):
            st.dataframe(df)

        with st.expander("Data Shape"):
            st.dataframe(df.shape)

        with st.expander("Data Type"):
            st.write(df.dtypes)
            
        with st.expander('Null'):
            st.dataframe(df.isnull().sum())

    elif submenu=='Statistik':
        st.write('Statistik')
        with st.expander('Describe'):
            st.dataframe(df.describe())
        
        with st.expander('Median'):
            st.dataframe(df.groupby('species').median())
        
        with st.expander('Matrix correlation'):
            corr = df.corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr,  annot=True, ax=ax, annot_kws={"size":5})
            ax.set_title('Matriks Correlation')
            plt.xticks(rotation=80)
            st.pyplot(fig)
        
        with st.expander('Pair Plot'):
            fig, ax = plt.subplots()
            sns.set_context('talk')
            sns.pairplot(df, hue='species')
            st.pyplot(fig)
        
        with st.expander('Scatter Plot'):
            fig, ax = plt.subplots(figsize=(7,7))
            plt.title('scatter plot sepal')
            sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='species')
            st.pyplot(fig)
        
        with st.expander('Scatter Plot 2'):
            fig, ax= plt.subplots(figsize=(7,7))
            plt.title('scatter plot petal')
            sns.scatterplot(data=df, x='petal_length', y='petal_width', hue='species')
            st.pyplot(fig)
        
        with st.expander('Box Plot'):
            fig, ax= plt.subplots(figsize=(7,7))
            sns.boxplot(data=df, orient='h')
            plt.title('Boxplot Chart')
            st.pyplot(fig)

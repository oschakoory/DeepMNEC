from PIL import Image
import streamlit as st
import matplotlib as plt
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import pandas as pd
import numpy as np
import tensorflow as tf
from streamlit_shap import st_shap
import shap
from shap import Explanation
from shap.plots import waterfall

tf.compat.v1.disable_v2_behavior()  # <-- HERE !
# tf.enable_eager_execution()

# CSS
def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_background():
    #bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-color: #ffffff;
    background-blend-mode: lighten;
    background-size: cover;
    }
    .css-b7s55g{
        background: #ca7ebf;
    }

    .css-1yjuwjr{
        font-size: 16px;
        font-weight:700;
        padding-left: 0.5rem;
    }

    .st-au {
    background-color: #c8e0d7;
    color: #f3f9fd;
    text-align: center;

    .main-svg{
        background-color: #11ffee00 !important;
    }

    </style>
    ''' 
    st.markdown(page_bg_img, unsafe_allow_html=True)


change_text = """
<style>
div.st-cs.st-c5.st-bc.st-ct.st-cu {visibility: hidden;}
div.st-cs.st-c5.st-bc.st-ct.st-cu:before {content: "Sélectionner l'information à afficher :"; visibility: visible;}
</style>
"""



########################################################################################################################################

# load logo
logo = Image.open('datas/logo_NEC.png')

set_background()

col1, mid, col2 = st.columns([2, 2, 8])
with mid:
    st.image(logo)
with col2:
    st.markdown("<h1 style='color: #c8e0d7;'>DeepMNEC</h1>",
                unsafe_allow_html=True)

# Loading data....
DOL = ('DOL_0-9',
'DOL_10-19',
'DOL_100-199',
'DOL_20-29',
'DOL_200-299',
'DOL_30-39',
'DOL_300-399',
'DOL_40-49',
'DOL_400-499',
'DOL_50-59',
'DOL_500-599',
'DOL_60-69',
'DOL_70-79',
'DOL_80-89',
'DOL_90-99')

Gender = ('gender_female','gender_male')

Gestational_age = ('gestational_age_23',
'gestational_age_24',
'gestational_age_25',
'gestational_age_26',
'gestational_age_27',
'gestational_age_28',
'gestational_age_29',
'gestational_age_30',
'gestational_age_31',
'gestational_age_32')


Birth_mode =('birth_mode_C-section', 'birth_mode_Vaginal')

Weight = ('Weight_1000-1099',
'Weight_1100-1199',
'Weight_1200-1299',
'Weight_1300-1399',
'Weight_1400-1499',
'Weight_1500-1599',
'Weight_1600-1699',
'Weight_1700-1799',
'Weight_1800-1899',
'Weight_2000-2099',
'Weight_2100-2199',
'Weight_2200-2299',
'Weight_2800-2899',
'Weight_500-599',
'Weight_600-699',
'Weight_700-799',
'Weight_800-899',
'Weight_900-999')


uploaded_file = st.sidebar.file_uploader("File uploader")

# selectbox for DOL
DOLselect = st.sidebar.selectbox("DOL", DOL)

# selectbox for Gender
Genderselect = st.sidebar.selectbox("Gender", Gender)

# selectbox for GA
GAselect = st.sidebar.selectbox("Gestational age", Gestational_age)

# selectbox for birth mode
BMselect = st.sidebar.selectbox("Birth mode", Birth_mode)

# selectbox for Weight
Weightselect = st.sidebar.selectbox("Weight", Weight)

# checkbox to display different options
cbx_proba = st.sidebar.button('Predict')

########################################################################################################################################

loaded_model = tf.keras.models.load_model('models/best-model_2pub.h5')
loaded_model.summary()

y = ['Control', 'NEC']
encoder = LabelEncoder()
encoder.fit(y)
        

def my_prediction(unknown, new_samp):

    # intialise data of lists.
    data = {0:[DOLselect,Genderselect,GAselect,BMselect,Weightselect],
            1:[1,1,1,1,1]}
    
    # Create DataFrame
    df = pd.DataFrame(data)

    new_samp = new_samp[["Species", "Relative_abundance"]]
    # group by similar taxa and sum abundance
    new_samp = new_samp.groupby('Species').sum('Relative_abundance')
    new_samp_ = new_samp.reset_index()
    new_samp_[['Species']] = new_samp_[['Species']].replace(
        's__', '', regex=True)  # remove s__ from species
    new_samp_[['Species']] = new_samp_[['Species']].replace(
        '_', ' ', regex=True)  # replace _ by \s
    new_samp_.rename(columns={'Species': 0}, inplace=True)
    new_samp_.rename(columns={'Relative_abundance': 1}, inplace=True)

    # -------------------------------------------------------------------------

    # concatenate species abundance + clinical data
    sample_unknown = pd.concat([new_samp_, df], axis=0)
    # re-order samples columns as training data
    features = pd.read_csv("datas/features.txt", header=None)
    features['value'] = 0
    features.rename(columns={'value': 1}, inplace=True)
    # remove all species not used during training
    sample_unknown_ = sample_unknown[sample_unknown[0].isin(features[0])]
    # remove all species present in unknown sample
    features_ = features[~features[0].isin(sample_unknown_[0])]
    # concatenate all species used in training to unknown samples
    test_sample = pd.concat((sample_unknown_, features_), axis=0)
    test_sample = test_sample.set_index(0)
    # reorder species column as training set
    test_sample_ = test_sample.reindex(index=features[0])
    test_sample_ = test_sample_.reset_index()
    test_sample_.rename(columns={1: unknown}, inplace=True)
    test_sample_ = test_sample_.set_index(0)
    test_sample_.index.names = [None]
    test_sample_[unknown] = test_sample_[unknown].astype(str).astype(float)

    # -------------------------------------------------------------------------

    # final prepapration before prediction
    test_sample_t = test_sample_.T

    sample = test_sample_t.to_numpy()

    prediction = np.argmax(loaded_model.predict(sample), axis=-1)

    prediction_ = np.argmax(to_categorical(prediction), axis=1)
    prediction_ = encoder.inverse_transform(prediction_)

    st.info(
        f"The predicted phenotype of {unknown} is: {prediction_[0]}")

    # -------------------------------------------------------------------------

    #get features from shap
    weights = pd.read_csv("datas/explainerSHAP.csv", sep=',', index_col=0, header=0)
    weights = weights.to_numpy()

    explainer = shap.DeepExplainer(loaded_model, weights)
    shap_values = explainer.shap_values(sample)


    #plot graphs

    # compute SHAP values

    st.info("Features importance explaining local prediction")

    st_shap(shap.summary_plot(shap_values[1],
                    feature_names=test_sample_t.columns,
                    plot_type="bar", show=False))
    
    st_shap(waterfall(Explanation(shap_values[1][0], 
        explainer.expected_value[1], 
        data=test_sample_t.iloc[0], 
        feature_names=test_sample_t.columns)))

if uploaded_file is not None:
    new_samp = pd.read_csv(uploaded_file, sep='\t')

    st.info(uploaded_file.name.split('_')[0])

    st.write(new_samp)

if cbx_proba:

    my_prediction(uploaded_file.name.split('_')[0], new_samp)

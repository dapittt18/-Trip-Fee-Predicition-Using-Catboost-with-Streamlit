# Import Libraries
import streamlit as st
import numpy as np
import pandas as pd
from joblib import load

# View wider web page
st.set_page_config(page_title = 'Cek Ongkos Kirim', layout="centered")

# Web header
html_temp = """ 
    <div style ="background-color:yellow;padding:20px"> 
    <h1 style ="color:black;text-align:center;">Cek Harga Ongkos Pengiriman</h1> 
    </div> 
    """
     
# Markdown
st.markdown(html_temp, unsafe_allow_html = True)


st.info("!This is the beta version of Logee Trans Trip Fee Prediction Apps!")

# Load best model
catboost = load('catboost_reg.z')

# Load encoder object
encoder_item = load('encoder1.z')
encoder_vehicle = load('encoder2.z')


vehicle_grup = ('Blind Van','CDD Bak','CDD Box','CDD Chiller',
       'CDD Long Bak','CDD Long Box','CDD Los Bak','CDD Wingbox','CDE Bak','CDE Box','Fuso Bak',
       'Fuso Box','Fuso Jumbo','Fuso Tronton','High Bed','Low Bed','Pick Up Bak','Pick Up Box','Self Loader',
       'Trailer Chiller Container 40 Feet','Trailer Dry Container 20 Feet','Trailer Dry Container 40 Feet',
       'Tronton Bak','Tronton Box','Tronton Dump Truck','Tronton Wingbox')

distance = st.number_input('Jarak (km)',0.0,5000.0, key = 'distance')
itemWeight = st.number_input('Berat Barang (ton)',0.0,101.0, key = 'itemWeight')
itemPackage = st.selectbox("Jenis Barang",['Bags', 'Cartons', 'CBM', 'CBN', 'Drum', 'Palet'], key = 'itemPackage')
vehicleGroupName = st.selectbox("Jenis Kendaraan",vehicle_grup, key = 'vehicleGroupName')

# Create new dataframe with the exact same column name on modelling stage
df = pd.DataFrame([{'itemWeight' : itemWeight, 'distance': distance, 'vehicleGroupName':vehicleGroupName, 'itemPackage':itemPackage}])

# Transform categorical input (truck type) with loaded encoder
df[encoder_vehicle.get_feature_names()] = encoder_vehicle.transform(df[['vehicleGroupName']].to_numpy().reshape(1, -1))

# Transform categorical input (item type) with loaded encoder
df[['x1_Bags','x1_Cartons','x1_CBM','x1_CBN','x1_Drum','x1_Palet']] = encoder_item.transform(df[['itemPackage']].to_numpy().reshape(1, -1))

# Delete categorical column vehicleGroupName dan itemPackage
df = df.drop(['vehicleGroupName','itemPackage'], axis = 1)

# Prediction
if st.button('Check'):
    prediction = catboost.predict(df)[0]
    st.write(f'Rp. {round(prediction)}')
    st.success("Success!")

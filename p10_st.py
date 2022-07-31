import streamlit as st
import pandas as pd
import numpy as np

header = st.container()
dataset = st.container()
loading1 = st.container()
loading2 = st.container()

with header:
    st.title('Détectez des faux billets')
    
with dataset:
    st.header('Fichier d’exemple')
    st.text('1500 billets selon six informations géométriques et leur nature')
    st.text('l\'algorithme de régression logistique est entraîné d\'après ce fichier')
    
    billets = pd.read_csv("billets_sortie.csv",sep=',')
    billets.is_genuine.replace({True: 'Vrai', False: 'Faux'},inplace=True)
    st.dataframe(data=billets, width=800, height=200)

from sklearn.linear_model import LogisticRegression
X_train = pd.read_csv("X_train.csv",sep=',')
y_train = pd.read_csv("y_train.csv",sep=',')
y_train.replace({True: 'Vrai', False: 'Faux'},inplace=True)
clf = LogisticRegression(random_state=0).fit(X_train, y_train)


with loading1:
    st.header('Charger des données:')
    st.text('Vous pouvez charger vos données ici (fichier csv) pour détecter la nature des')
    st.text('billets (vrai/faux)')
    sel_col, disp_col = st.columns(2)

    uploaded_file = sel_col.file_uploader("Charger un fichier")
    if uploaded_file is not None:
        billets_production = pd.read_csv(uploaded_file)
        billets_production.set_index(billets_production.id, inplace=True)
        billets_production = billets_production[['diagonal', 'height_left', 'height_right','margin_low', 'margin_up','length']]
        
        results_rl = billets_production.copy()
        results_rl['genuine'] = clf.predict(billets_production)

        disp_col.text('Nature des billets')
        disp_col.write(results_rl.genuine)


with loading2:
    st.header('Entrer des données:')
    st.text('Vous pouvez entrer vos données ici pour détecter la nature d\'un billet (vrai/faux)')
    
    sel_col, disp_col = st.columns(2)
    diagonal = sel_col.text_input('diagonal')
    height_left = sel_col.text_input('height_left')
    height_right = sel_col.text_input('height_right')
    margin_low = sel_col.text_input('margin_low')
    margin_up = sel_col.text_input('margin_up')
    length = sel_col.text_input('length')
    
    if length is not '':
        billet_pred = pd.DataFrame(np.array([[diagonal, height_left, height_right, margin_low, margin_up, length]]), columns = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length'])
        billet_pred['genuine'] = clf.predict(billet_pred)
        disp_col.text('Nature du billet')
        disp_col.write(billet_pred.genuine[0])

import streamlit as st
from PIL import Image
import os
import pandas as pd
import numpy as np
import glob
import cv2
import random
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from numpy import asarray
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import RMSprop, Adam
​
​
#modify the name of the app
st.set_page_config(page_title = "Pneumonie detection")
st.set_option('deprecation.showPyplotGlobalUse', False)
#IMPORT DES DONNEES
lst_img_train_normal = glob.glob('/Users/samuelchemama/projet_fin_batch/chest_xray/data/train/normal/*.jpeg')
lst_img_train_pneumonia = glob.glob('/Users/samuelchemama/projet_fin_batch/chest_xray/data/train/pneumonia/*.jpeg')
lst_img_test_normal = glob.glob('/Users/samuelchemama/projet_fin_batch/chest_xray/data/test/normal/*.jpeg')
lst_img_test_pneumonia = glob.glob('/Users/samuelchemama/projet_fin_batch/chest_xray/data/test/pneumonia/*.jpeg')
lst_img_val_normal = glob.glob('/Users/samuelchemama/projet_fin_batch/chest_xray/data/val/normal/*.jpeg')
lst_img_val_pneumonia = glob.glob('/Users/samuelchemama/projet_fin_batch/chest_xray/data/val/pneumonia/*.jpeg')
# we put all the information in a list and we put a target (0 : normal , 1 :pneumonia)
#TRAIN DF :
lst_train_normal = [[x,0] for x in lst_img_train_normal]
lst_train_pneumonia = [[x,1] for x in lst_img_train_pneumonia]
lst_train = lst_train_normal + lst_train_pneumonia
random.shuffle(lst_train)
train = pd.DataFrame(lst_train,columns = ['path','target'])
#TEST DF
lst_test_normal = [[x,0] for x in lst_img_test_normal]
lst_test_pneumonia = [[x,1] for x in lst_img_test_pneumonia]
lst_test = lst_test_normal + lst_test_pneumonia
random.shuffle(lst_test)
test = pd.DataFrame(lst_test,columns = ['path','target'])
#VAL DF
lst_val_normal = [[x,0] for x in lst_img_val_normal]
lst_val_pneumonia = [[x,1] for x in lst_img_val_pneumonia]
lst_val = lst_val_normal + lst_val_pneumonia
random.shuffle(lst_val)
val = pd.DataFrame(lst_val,columns = ['path','target'])
​
##AFFICHAGE
# we take randomly 10 images in the train dataset for normal and pneumonie
rd_lst_normal = random.sample(lst_img_train_normal,1)
rd_lst_pneumonia = random.sample(lst_img_train_pneumonia,1)
# on concat 2 list
lst_rd = rd_lst_normal + rd_lst_pneumonia
#we shuffle the list (in place method)
random.shuffle(lst_rd)
​
​
#fonction modele xgboost
def model(upload_file):
    res = []
    img = Image.open(upload_file)
    img = img.resize((225,225),Image.ANTIALIAS) #resize image
    img = asarray(img)
    img = img/255
    #st.image(img)
    img = img.reshape(1,img.shape[0]*img.shape[1])
    #st.image(img)
    y_pred = loaded_model.predict(img)
    res.append(y_pred)
    y_pred_fi = np.array(res)
    return y_pred_fi
​
def model_cnn(upload_file):
    img = Image.open(upload_file)
    img = img.resize((225,225),Image.ANTIALIAS)
    img = asarray(img)
    img = img/255
    img = img.reshape(1,img.shape[0],img.shape[1],1)
    y_hat = loaded_model.predict(img)
    y_hat = np.argmax(y_hat,axis=1)
    return y_hat
​
​
# SideBar selectbox on the left
select = st.sidebar.selectbox('Selectionnez  : ',['Accueil',"Dataset","Architecture CNN","Modelisation"])
​
​
if select == 'Accueil':
​
    st.title("Détection de pneumonie chez l'enfant")
    st.header("L'objectif de ce projet est d'assister les médecins lors de leurs diagnostics de pneumonie infantile")
    st.subheader("Nous avons pour cela entrainé un modèle sur un set de donnée de poumons sains & malades en haute résolution.")
    st.subheader('Le modèle à été entrainé sur plus de **5000 images**')
    st.markdown(
    """<a href="https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia">Pneumonie Dataset</a>""", unsafe_allow_html=True,
)
    img = Image.open('pneumonie.jpg')
    st.image(img)
​
if select == "Dataset":
    select_df = st.selectbox('Selectionnez un dataset:', ['Train','Test'])
​
    if select_df == 'Train':
​
        #Affichage countplot Train
        st.subheader('**Train Dataset**')
        st.markdown('Nous pouvons constaté que les données sont fortements non balancées, cette information à été prise en compte lors de la construction de nos modèles')
        fig1 = sns.countplot(data = train , x = 'target',palette=['#432371',"#FAAE7B"])
        plt.title('Train DataSet')
        st.write(fig1)
        st.pyplot()
        st.markdown("Nous affichons quelques images brut que nous avons dans notre jeu de données d'entrainement")
​
        #Affichage images poumons
        fig2 = plt.figure(figsize = (15,15))
        for i in range(len(lst_rd)):
            plt.subplot(3,3,i+1)
​
            if 'normal' in lst_rd[i]:
                img = cv2.imread(lst_rd[i])
                plt.imshow(img)
                plt.title('NORMAL')
            else:
                img = cv2.imread(lst_rd[i])
                plt.imshow(img)
                plt.title('PNEUMONIE')
​
        st.write(fig2)
        st.pyplot()
​
​
​
    elif select_df == 'Test':
        st.markdown('Le jeu de donnée de test est plus équilibré')
        fig3 = sns.countplot(data = test , x = 'target',palette=['#432371',"#FAAE7B"])
        plt.title('Test DataSet')
        st.write(fig3)
        st.pyplot()
​
if select == 'Architecture CNN':
    st.title('Comment fonctionne les CNN?')
    st.header('I) Objectif')
    st.markdown("Les CNN (Réseau de Neurones convolutifs) sont impliqués dans la plupart des algorithmes de Computer Vision.\
    Ils sont notamment utiles pour la classification d'images . L'idée derrière cette architecture est que l'on puisse déterminer des patterns dans\
    une image, là ou avec un simple reséau de neurones (MLP) cette tâche n'était pas réalisable . En effet, lorsque nous appliquons un simple MLP\
    à une image nous devons aplatir l'image en autant de features (pixels) qui la compose . De ce fais ces features sont considérés comme indépéndantes\
    par ce type de modele : Une problématique que permet de résoudre les CNN.\n Nous verrons ici une première approche pour débutant au CNN")
    st.header('II) Fonctionnement')
    st.markdown("L'architecture d'un CNN ressemble à cela : ")
    img = Image.open('architecture-cnn-fr.jpg')
    img = img.resize((800,200),Image.ANTIALIAS)
    st.image(img)
    st.subheader("A) Couche de convolution")
    st.markdown("La couche de convolution est le coeur de ces algorithmes. **Un point important** : Pour un ordinateur, une image n'est rien d'autre\
    qu'une matrice de pixels. L'objectif est de faire passer un kernel sur l'image en input afin de créer une représentation abstraite de l'image initiale.\
    Cette image abstraite représentera des patterns de l'image initiale. Nous pouvons donc régler la taille de ce kernel F que l'on fait passé sur notre \
    image, le décalage avec lequel on déplace notre kernel est appelé le 'Stride'. Pour une image de taille I X I , un stride de taille S , une taille de kernel de F x F \
    l'image convonlutionné est de taille  O x O: ")
    img = Image.open('CodeCogsEqn.png')
    st.image(img)
    st.markdown("Nous pouvons voir ci dessous un filtre F appliqué à une image en input : ")
    st.image('convolution_animated.gif')
    st.markdown("Dans cet exemple avons une image de taille 5x5 , un stride de taille 1 , et nous appliquons un kernel de taille 3x3 : Nous avons bien en \
    sortie une image de taille O = (5-3)/1 +1 = 3 de côté \
    L'image de sortie est le résultat de l'application de notre kernel , ici notre kernel est la matrice : ")
    img = Image.open('CodeCogsEqn-2.gif')
    st.image(img)
    st.markdown("Cependant il peut arrivé que nous ne souhaitons pas réduire la taille des matrices de sorties : Pour cela nous pouvons appliquer ce qu'on appel du **Padding**\
    cela permet de remplir notre image de 0 sur les bords afin d'augmenter sa dimension initial pour avoir en sortie d'une couche de convolution une image de même taille que l'image initiale.")
    st.markdown("Une question se pose : Mais **comment savoir qu'elle kernel utiliser?**. Eh bien c'est notre réseau de neurone qui va l'apprendre seul.\
    Initialement nos avons un kernel rempli de poids initialisés aléatoirement, puis lors de l'entrainement du modèle il va modifier ces poids afin de minimiser\
    la fonction d'erreur : Tout comme un MLP classique.\
    **PS**: Si vous n'avez pas de connaissance sur le fonctionnement d'un MLP et des étapes qui le compose je vous invite à suivre ce lien : ")
    st.markdown(
    """<a href="https://missinglink.ai/guides/neural-network-concepts/perceptrons-and-multi-layer-perceptrons-the-artificial-neuron-at-the-core-of-deep-learning/">Concept des MLP</a>""", unsafe_allow_html=True,
)
    st.subheader("B) Pooling")
    st.markdown("Un autre couche fréquemment utilisée est la couche de Pooling : Il existe plusieurs sortes de pooling nous verrons nous concernant le **MaxPooling**\
     et **AveragePooling**. L'idée du pooling est de diminuer la taille de nos images .L'opération de pooling consiste à réduire la taille des images, tout en préservant leurs caractéristiques importantes.\
     Pour cela, on découpe l'image en cellules régulière, puis on garde au sein de chaque cellule la valeur maximale..\
     Dans le cas du MaxPooling nous gardons la valeure maximal du pixel dans le sous ensemble, dans le cas du AveragePooling nous faisons une moyenne des pixels de la zone de l'image \
     Dans l'image ci dessous nous voyons qu'en appliquant un pooling_size de (2,2) et un stride de 2 nous obtenons une images de sorties réduites")
    img = Image.open('Maxpooling.png')
    st.image(img)
​
    st.subheader("C) La Couche Fully Connected")
    st.markdown("La couche Fully Connected constitue la dernière couche du CNN (et pas que du CNN). Comme son nom l'indique tous les neurones de cette couche sont complètement connectés aux features de la couche\
    précédente. Ainsi nous devons aplatir les images de sorties des couches précédentes (flatten) pour pouvoir connectés tous les pixels à cette dernière couche. Nous ajoutons de plus un layer d'output\
    avec une fonction d'activation dépendente de l'objectif visé. Par exemple dans un cas de classification binaire d'une image l'output de sortie sera un unique neurone ayant pour fonction d'activation\
    la fonction sigmoide . Ce neurone retournera la probabilité d'appartenance à chaque classe.")
​
    st.subheader("Transfert Learning")
    st.markdown("Un CNN est un empilement de couche de convolution et de pooling avec différents hyperparamètres (nombre de kernel, taille du kernel,padding etc..)\
    plus on empile de couche plus le modèle sera profond. Comme évoqué plus haut, en dernière couche nous introduisons une couche Dense Fully Connected.\
    Un point est cependant problématique : Lorsqu'un réseau est profond le nombre de poids à apprendre est **CONSIDERABLE** , il faut donc des machines très puissantes pour pouvoir\
    entrainer des modèles très performants . Et c'est là que le **Transfert Learning** rentre en jeu. Son principe est d'utiliser des modèles déjà pré-entrainés sur des millions d'images,\
    et récupérer les poids associés à ces modèles . Puis on 'fine tune' le modèle c'est à dire qu'on modifie la couche de sortie (en fonction du nombre de catégorie que l'on souhaite prédire).\
    Nous utilisons pour ce projet le modèle VGG16 dont les éléments seront présentés dans la partie **Modélisation**.")
​
if select == 'Modelisation':
    select4 = st.sidebar.selectbox('Selectionnez un modèle : ',["CNN(recommandé)","Xgboost","Random Forest Classifier"])
​
    if select4 == "Random Forest Classifier":
​
        if st.button('Affichez les performances du modèle sur le jeu de Test'):
             img = Image.open('confusion_matrix_rf.png')
             st.image(img)
             st.markdown("Le modèle à une accuracy de **74%**, il est très bon sur le recall de la classe 1 (**99%**)\
             en revanche le modèle prédit souvent qu'une personne à une pneumonie quand ce n'est pas le cas (recall de **37%** sur la classe 0).")
​
        filename = '/Users/samuelchemama/projet_fin_batch/chest_xray/data/RandomForest.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        st.header('Veuillez rentrer une image radiographique de poumon : ')
        upload_files = st.file_uploader('', accept_multiple_files=True)
        for upload_file in upload_files:
            st.markdown("La prédiction du modèle est : ")
            if model(upload_file) == 1:
                 "Pneumonie"
            else:
                "Normal"
            img = Image.open(upload_file)
            img = img.resize((225,225),Image.ANTIALIAS)
            st.image(img)
​
    if select4 == 'Xgboost':
        if st.button('Affichez les performances du modèle sur le jeu de Test'):
            img = Image.open('confusion_matrix_xgboost.png')
            st.image(img)
            st.markdown("Le modèle à une accuracy de **74%**, il est très bon sur le recall de la classe 1 (**99%**)\
            en revanche le modèle prédit souvent qu'une personne à une pneumonie quand ce n'est pas le cas (recall de **31%** sur la classe 0).")
​
​
        filename = '/Users/samuelchemama/projet_fin_batch/chest_xray/data/test_xgboost.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        st.header('Veuillez rentrer une image radiographique de poumon : ')
        upload_files = st.file_uploader('', accept_multiple_files=True)
        for upload_file in upload_files:
            st.markdown("La prédiction du modèle est : ")
            if model(upload_file) == 1:
                "Pneumonie"
            else:
                "Normal"
            img = Image.open(upload_file)
            img = img.resize((225,225),Image.ANTIALIAS)
            st.image(img)
​
    if select4 == 'CNN(recommandé)':
        #display confusion matrix
        if st.button('Affichez les performances du modèles sur le jeu de Test'):
            img = Image.open('confusion_matrix_cnn.png')
            st.image(img)
​
            #on recupere df_cnn_classification_report
            df_cnn = pd.read_pickle("/Users/samuelchemama/projet_fin_batch/chest_xray/data/df_cnn.pkl")
            st.write(df_cnn)
​
        #load model
        json_file = open('/Users/samuelchemama/projet_fin_batch/chest_xray/data/model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("/Users/samuelchemama/projet_fin_batch/chest_xray/data/model_cnn.h5")
        optimizer = Adam(lr=0.0001, decay=1e-5)
        loaded_model.compile(optimizer = optimizer, metrics = ['accuracy'],loss = 'categorical_crossentropy')
​
        #apply model to the image download
        st.header('Veuillez rentrer une image radiographique de poumon : ')
        upload_files = st.file_uploader('', accept_multiple_files=True)
        for upload_file in upload_files:
            st.markdown("La prédiction du modèle est : ")
            if model_cnn(upload_file) == 1:
                "Pneumonie"
            else:
                "Normal"
            img = Image.open(upload_file)
            img = img.resize((225,225),Image.ANTIALIAS)
            st.image(img)


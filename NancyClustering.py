#importing packages
import pandas as pd
import numpy as np
import sklearn # Package for machine learning tools
import matplotlib.pyplot as plt # package for 2D charts
import streamlit as st # package for creating web platform
import seaborn as sns # package for generating charts
from streamlit_option_menu import option_menu #package for creating pages
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage

st.title("Unsupervised Learning Project")

#Global level
#Importing data into project
dataset=pd.read_csv("/Users/nancyhayford/Documents/TrialProject1/Country-data.csv")

#selectig attributes for analysis
data=dataset[['child_mort', 'exports', 'health','income','inflation', 'life_expec','total_fer','gdpp']]

#function for creating navigation
def page1():
    st.header("Load data")
    st.write(data)
    # upload file into project using file uploader
    mydata = st.file_uploader("Upload file here", type=["csv", "txt"])
    if mydata is not None:
        st.success("file uploaded sucessfully")

    if mydata is not None:
        datum = pd.read_csv(mydata)
        st.write(datum)

def page2():
    st.header("Exploratory Data Anlysis")
    st.write(data.describe())


def page3():
    st.header("Visualisation")
    cor=data.corr()

    #heatmap
    fig,ax=plt.subplots()
    sns.heatmap(cor, square=True, annot=True)
    st.pyplot(fig)

def page4():
    #KMeans clustering
    st.subheader("K-Mean Clustering")
    n_clusters=3
    random_state=40
    Kmeans=KMeans(n_clusters=n_clusters, random_state=random_state)
    Kmeans.fit(data)

    #add the cluster label to the table
    dataset['K-Means'] = Kmeans.labels_
    st.write(dataset)

    #KMeans algorithm performance
    st.subheader("Evaluation of the K means algorithm")
    sil=silhouette_score(data, Kmeans.labels_)
    st.write("The K means performance is:",sil*100,'%')

    #Generate a dendogram
    linkage_matrix=linkage(data,method='ward')
    dendrogram(linkage_matrix)

    fig2, ax2 = plt.subplots()
    plt.title("dendogram")
    plt.xlabel("sample")
    plt.ylabel("Distance")
    st.pyplot(fig2)

    #visualise the clusters
    st.subheader("Visualising the clusters using K-Means Algorithm")
    fig1, ax1= plt.subplots()

    #scatter plot
    plt.scatter(data['income'],data['inflation'], c= dataset['K-Means'], cmap='viridis')
    plt.scatter(Kmeans.cluster_centers_[:,3], Kmeans.cluster_centers_[:,4], c='red', label="Centroids")
    plt.title("K-Means Clustering")
    plt.xlabel('Income')
    plt.ylabel('Inflation')
    plt.legend()
    st.pyplot(fig1)

    #Agglomearive
    st.subheader(" Generating Agglomerative Clustering")
    Agg=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward')
    Agg.fit_predict(data)
    dataset['Agglomerative'] =  Agg.labels_
    st.write(dataset)

def page5():
    st.header("Principal Component Analysis")
    #z-score/normalise/Scaler
    scaler=StandardScaler()
    scaler.fit(data)
    x=scaler.transform(data)
    st.write(x)

    #PCA Model
    n=X.shape[1]
    pca=PCA(n_components=n, random_state=1)
    X_new=pca.fit_transform(x)

    #Visualizing two PCAs
    figs,axes=plt.subplots(1,2)
    axes[0].scatter(X[:,0],X[:,1],c=target)
    axes[0].seth_xlabel('X1')
    axes[0].seth_ylabel('X2')
    axes[0].seth_title('Original scatered data')

    axes[1].scatter(X_new[:,0],X[:,1],c=target)
    axes[1].seth_xlabel('PCA1')
    axes[1].seth_ylabel('PCA2')
    axes[1].seth_title('After PCA')

    st.pyplot(figs)

    #Computing for the explained variance individual components
    pc_comps=['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8']
    exp_var=pca.explained_variance_ratio_
    variance_pca=pd.DataFrame(np.round(exp_var, 3),index=pc_comps,Columns=['Explained variance'])
    variance_pca = pd.DataFrame(np.round(exp_var, 3), index=pc_comps, columns=['Explained variance '])
    st.write('Explained variance are:',variance_pca)

def page6():
    st.header("Association Rules Mining")

pages= {
    'Loading data': page1,
    'Exploratory': page2,
    'Visualisation': page3,
    'Clustering': page4,
    'Dimensionlity': page5,
    'Association Rules mining': page6
}

st.sidebar.header("Clustering")
selected_page=st.sidebar.selectbox("Select a page",list(pages.keys()))

#display pages
pages[selected_page]()
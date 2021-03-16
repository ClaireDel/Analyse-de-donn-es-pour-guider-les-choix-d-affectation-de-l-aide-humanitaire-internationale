import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import scipy.cluster.hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.mixture import GaussianMixture as GMM
from sklearn.decomposition import PCA
from matplotlib.collections import LineCollection
#------------------------------------------------------------------------------


# Données test

G20 = pd.read_csv('g2-2-20.txt', sep='    ', engine='python')
G100 = pd.read_csv('g2-2-100.txt', sep='   ', engine='python')
jain = pd.read_csv('jain.txt', sep='\t')
AG = pd.read_csv('Aggregation.txt', sep='\t')
PA = pd.read_csv('pathbased.txt', sep='\t')




#------------------------------------------------------------------------------


# Données réelles




# On note D le tableau initial, non modifié. Le garder intact permet de se référer
# aux vraies valeurs lors du processur de réflexion.

# On note D1 le tableau 'utile', corrigé et normalisé. C'est celui-ci qui sera
# employé dans la première partie de la classification.

# On note D3 le tableau initial avec modifications de valeurs aberrantes. Ce
# tableau permet de suivre le raisonnement effectué avec après réduction du nombre
# de pays.


# Par ailleurs, on note : 
    # Dc le DataFrame de corrélation des variables
    # Dd le DataFrame descriptif des données de D
    



# Importation du jeu de données
D = pd.read_csv('data.csv')


# Analyse initiale 
Dc = D.corr()
Dd = D.describe()


# Création de D1 et modification des valeurs aberrantes
D1 = D.copy()
D1.iloc[7,9] = 54907
D1.iloc[158,9] = 42300
D1.iloc[159,9] = 65118
D1.iloc[12,7] = 72
D1.iloc[54,8] = 1.92
D1.iloc[112,8] = 7
D1.iloc[75,9] = 34483
D1.iloc[114,9] = 81697


# Création de D3 pour la partie finale, qui contient encore les valeurs réelles
# et les noms de pays
D3 = D1.copy() 


# Suppression des noms de pays pour permettre la réduction des données
del(D1['country'])

# Réduction des données
D1 = StandardScaler().fit_transform(D1)




def dbscan(X,e,m):
    db = DBSCAN(eps=e,min_samples=m)
    db.fit(X)
    Y = db.fit_predict(X)
    C = []
    B = []
    for k in range (max(Y)+1) :
        C.append([])
    for i in range(len(Y)) :
        if Y[i] != -1 :
            C[Y[i]].append(D.iloc[i,0]) 
        if Y[i] == -1 :
            B.append(D.iloc[i,0])
    # plt.scatter(X[:,0],X[:,4],c=Y,s=10)
    # plt.title('DBSCAN')
    return B
  
  
def kmeans(X,K) :
    kmeans = KMeans(n_clusters=K, init='k-means++', n_init=100)
    kmeans.fit(X)
    Y = kmeans.predict(X)
    C = []
    for k in range (max(Y)+1) :
        C.append([])
    for i in range(len(Y)) :
        C[Y[i]].append(D.iloc[i,0]) 
    plt.scatter(X[:,0], X[:,1], c=Y, s = 10)
    plt.title('K-means')
    return C
 
   
def cah(X,t) :
    Z = linkage(X,method='ward',metric='euclidean')
    Y = scipy.cluster.hierarchy.fcluster(Z, t, criterion='distance')
    C = []
    for k in range (max(Y)) :
        C.append([])
    for i in range(len(Y)) :
        C[Y[i]-1].append(D.iloc[i,0]) 
    plt.scatter(X[:,0], X[:,1], c=Y, s = 10)
    plt.title('CAH')
    plt.figure(figsize=(20, 10))
    dendrogram(Z)
    return C

  
def gauss(X,K) :
    gmm = GMM(n_components=K, n_init=10, max_iter=1000)
    gmm.fit(X)
    Y = gmm.predict(X)
    C = []
    for k in range (max(Y)+1) :
        C.append([])
    for i in range(len(Y)) :
        C[Y[i]].append(D.iloc[i,0]) 
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=10)
    plt.title('Mélange des gaussiennes')
    plt.show()
    return C


def acp(X,n) :
    pca = PCA(n_components=n)
    pca.fit(X) 
    P = pca.transform(X)
    # pcs = pca.components_
    # display_circles(pcs, 3, pca, [(0,1),(0,2),(1,2)], labels = np.array(D1.columns))
    return P
    

def cah_reduit(t) :   #Fonction pour réaliser la CAH a partir d'un nombre de pays limité
    D2 = acp(D1.copy(),4)
    N = pd.Series.to_list(D['country'])
    # On crée un tableau D2 ne comportant plus que les pays intéressants
    for k in range(1,168) :
        if (N[len(N)-k] in A) == False :
            D2 = np.delete(D2,(len(N)-k), axis=0)
            D3.drop(D3.index[(len(N)-k)], inplace=True)
    Z = linkage(D2,method='ward',metric='euclidean')
    Y = scipy.cluster.hierarchy.fcluster(Z, t, criterion='distance')
    C = []
    for k in range (max(Y)) :
        C.append([])
    for i in range(len(Y)) :
        C[Y[i]-1].append(D3.iloc[i,0]) 
    plt.scatter(D2[:,0], D2[:,1], c=Y, s = 10)
    plt.title('CAH')
    plt.show()
    plt.figure(figsize=(20, 10))
    dendrogram(Z)
    return C

# -----------------------------------------------------------------------------

# Croisement des données, pour l'ACP finale réduite à 41 pays (même raisonnement
# pour la classification à 27 pays)

X1 = ['Afghanistan',
  'Angola',
  'Benin',
  'Botswana',
  'Burkina Faso',
  'Burundi',
  'Cameroon',
  'Central African Republic',
  'Chad',
  'Comoros',
  'Congo Dem. Rep.',
  'Congo Rep.',
  "Cote d'Ivoire",
  'Equatorial Guinea',
  'Eritrea',
  'Gabon',
  'Gambia',
  'Ghana',
  'Guinea',
  'Guinea-Bissau',
  'Haiti',
  'Iraq',
  'Kenya',
  'Kiribati',
  'Lao',
  'Lesotho',
  'Liberia',
  'Madagascar',
  'Malawi',
  'Mali',
  'Mauritania',
  'Micronesia Fed. Sts.',
  'Mongolia',
  'Mozambique',
  'Namibia',
  'Niger',
  'Nigeria',
  'Pakistan',
  'Rwanda',
  'Senegal',
  'Sierra Leone',
  'Solomon Islands',
  'South Africa',
  'Sudan',
  'Tajikistan',
  'Tanzania',
  'Timor-Leste',
  'Togo',
  'Uganda',
  'Vanuatu',
  'Venezuela',
  'Yemen',
  'Zambia']

X2 = ['Afghanistan',
  'Angola',
  'Benin',
  'Botswana',
  'Burkina Faso',
  'Burundi',
  'Cameroon',
  'Central African Republic',
  'Chad',
  'Comoros',
  'Congo Dem. Rep.',
  'Congo Rep.',
  "Cote d'Ivoire",
  'Equatorial Guinea',
  'Eritrea',
  'Gabon',
  'Gambia',
  'Ghana',
  'Guinea',
  'Guinea-Bissau',
  'Haiti',
  'Iraq',
  'Kenya',
  'Kiribati',
  'Lao',
  'Lesotho',
  'Liberia',
  'Madagascar',
  'Malawi',
  'Mali',
  'Mauritania',
  'Mozambique',
  'Namibia',
  'Niger',
  'Nigeria',
  'Pakistan',
  'Rwanda',
  'Senegal',
  'Sierra Leone',
  'Solomon Islands',
  'South Africa',
  'Sudan',
  'Tanzania',
  'Timor-Leste',
  'Togo',
  'Uganda',
  'Yemen',
  'Zambia']

X3 = ['Afghanistan',
  'Angola',
  'Benin',
  'Botswana',
  'Burkina Faso',
  'Burundi',
  'Cameroon',
  'Central African Republic',
  'Chad',
  'Comoros',
  'Congo Dem. Rep.',
  'Congo Rep.',
  "Cote d'Ivoire",
  'Equatorial Guinea',
  'Gambia',
  'Ghana',
  'Guinea',
  'Guinea-Bissau',
  'Haiti',
  'Iraq',
  'Kenya',
  'Kiribati',
  'Lao',
  'Lesotho',
  'Liberia',
  'Madagascar',
  'Malawi',
  'Mali',
  'Mauritania',
  'Micronesia Fed. Sts.',
  'Mozambique',
  'Namibia',
  'Niger',
  'Rwanda',
  'Senegal',
  'Sierra Leone',
  'Solomon Islands',
  'South Africa',
  'Tajikistan',
  'Tanzania',
  'Togo',
  'Uganda',
  'Zambia']

A = []
for x in X1 :
    if x in X2 and x in X3 :
        A.append(x)
        
        
#------------------------------------------------------------------------------


# Fonction pour afficher le cercle de corrélation des variables dans les plans de l'ACP
# (trouvée sur OpenClassrooms.com)

def display_circles(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None):
    for d1, d2 in axis_ranks: # On affiche les 3 premiers plans factoriels, donc les 6 premières composantes
        if d2 < n_comp:

            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(7,7))

            # détermination des limites du graphique
            if lims is not None :
                xmin, xmax, ymin, ymax = lims
            elif pcs.shape[1] < 30 :
                xmin, xmax, ymin, ymax = -1, 1, -1, 1
            else :
                xmin, xmax, ymin, ymax = min(pcs[d1,:]), max(pcs[d1,:]), min(pcs[d2,:]), max(pcs[d2,:])

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if pcs.shape[1] < 30 :
                plt.quiver(np.zeros(pcs.shape[1]), np.zeros(pcs.shape[1]),
                   pcs[d1,:], pcs[d2,:], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                lines = [[[0,0],[x,y]] for x,y in pcs[[d1,d2]].T]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        plt.text(x, y, labels[i], fontsize='14', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD  # Cambio de PCA a TruncatedSVD

def realizar_svd(datos, n_componentes=2):
    '''
    Precondición:
        datos es una lista de estructuras vectoriales.
        n_componentes es un entero mayor que 0. Por defecto n_componentes es 2
    Poscondición:
        devuelve una lista de estructuras vectoriales de n dimensiones
    '''
    svd = TruncatedSVD(n_components=n_componentes)
    datos_svd = svd.fit_transform(datos)
    return datos_svd


def entrenar_kmeans(datos, k):
    '''
    Precondición:
        datos es una lista de estructuras vectoriales.
        k es un número entero mayor que 0.
    Poscondición:
        devuelve una lista de etiquetas que representan los clusters asignados a cada instancia.
        devuelve también los centroides de cada cluster.
    '''
    # Crear un modelo de k-means
    modelo_kmeans = KMeans(n_clusters=k)

    # Ajustar el modelo a los datos
    modelo_kmeans.fit(datos)

    # Obtener las etiquetas de los clusters y los centros de los clusters
    etiquetas = modelo_kmeans.labels_
    centros = modelo_kmeans.cluster_centers_

    return etiquetas, centros


def visualizar_clusters(datos, etiquetas, centros):
    '''
    Precondición:
        datos es una lista de vectores de 2 dimensiones.
        etiquetas es una lista de números que representan la pertenencia de cada instancia a un cluster.
        centros son los centroides de cada cluster.
    Poscondición:
        se muestra un gráfico que representa instancias en forma de puntos y clusters en forma de colores.
    '''
    plt.scatter(datos[:, 0], datos[:, 1], c=etiquetas, cmap='viridis')
    plt.scatter(centros[:, 0], centros[:, 1], marker='X', s=200, linewidths=2, color='red')
    plt.title('Resultado del Clustering')
    plt.show()


def kmeans(datos, k):
    '''
    Precondición:
        datos es una lista de vectores. Cada vector pertenece a una instancia.
        k es un número entero mayor que 0.
    Poscondición:
        devuelve una lista de etiquetas que representan los clusters asignados a cada instancia.
    '''
    datos_svd = realizar_svd(datos, n_componentes=2)

    # Realizar clustering en el espacio reducido por PCA
    etiquetas, centros = entrenar_kmeans(datos_svd, k)

    # Visualizar los resultados
    visualizar_clusters(datos_svd, etiquetas, centros)

    return etiquetas


def buscarCodo(datos, kmin, kmax):
    '''
    Precondición:
        datos es una lista de estructuras vectoriales.
        kmin y kmax son enteros.
        0 < kmin < kmax
    Poscondición:

    '''
    inercia = []

    # barrer todos los K para encontrar un punto de codo
    for k in range(kmin, kmax+1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(datos)
        
        # almacenar la inercia para cada resultado del k-means
        inercia.append(kmeans.inertia_)

    # calcular el punto de codo
    deltas = np.diff(inercia, 2)
    punto_codo = np.argmax(deltas) + kmin

    k_optimo = punto_codo

    print(f"k_optimo = {k_optimo}")
    # obtener el modelo k-means con el número de clusters que el algoritmo estima como óptimo
    modelo_kmeans = KMeans(k_optimo)
    modelo_kmeans.fit(datos)
    etiquetas = modelo_kmeans.labels_

    return etiquetas
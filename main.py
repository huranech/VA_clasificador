import os
import sys
import getopt
import joblib
from matplotlib import pyplot as plt

import utiles
import clustering
import classifier
import numpy as np
import preprocessor
import pandas as pd
from sklearn.metrics import f1_score
from scipy.sparse import issparse


if __name__ == "__main__":

    input_args = sys.argv[1:]
    short_options = "i:o:d:s:p:h:"
    long_options = ['input=', 'output=', 'do=', 'select=', 'parameters=', 'help']

    try:
        opts, args = getopt.getopt(input_args, short_options, long_options)
    except getopt.GetoptError:
        print('uso: nombre_del_programa.py -i <archivo_entrada> -o <archivo_salida>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('uso: nombre_del_programa.py -i <archivo_entrada> -o <archivo_salida>')
            sys.exit()
        elif opt in ("-i", "--input"):
            input_file = arg
        elif opt in ("-o", "--output"):
            output_file = arg
        elif opt in ("-d", "--do"):
            command = arg
        elif opt in ("-s", "--select"):
            select = arg
        elif opt in ("-p", "--parameters"):
            parameters = arg


    if command == "preprocess":
        print("[*] preprocesando dataset...")
        # comprobar que los parametros son correctos
        if parameters not in ["bow", "tfidf", "we", "transformers"]:
            print("[!]Error: el parámetro -p sólo puede tomar los valores 'bow', 'tfidf', 'we' y 'transformers'")
            exit(0)
        if not os.path.exists(input_file):
            print(f"[!]Error: El archivo {input_file} no existe.")
            exit(0)

        # cargar el CSV
        df = pd.read_csv(input_file)

        # seleccionar la columna a la que se le va a someter a preproceso
        documentos_crudos = df[select]

        # preprocesar texto en función de la técnica de vectorización seleccionada
        documentos_preprocesados = preprocessor.preprocesar_texto(documentos_crudos)

        if parameters == "bow":
            documentos_preprocesados = preprocessor.bow(documentos_preprocesados)
        elif parameters == "tfidf":
            documentos_preprocesados = preprocessor.tfidf(documentos_preprocesados)
        elif parameters == "we":
            documentos_preprocesados = preprocessor.we(documentos_preprocesados)
        elif parameters == "transformers":
            documentos_preprocesados = preprocessor.transformers(documentos_preprocesados)

        # guardar estructura de datos en el directorio de trabajo
        joblib.dump(documentos_preprocesados, output_file)
        print("[*] Preproceso completado")
    

    elif command == "classify":
        print("[*] Tarea de clasificación en marcha")
        input_file = input_file.split(",")

        # obtener los datos de entrenamiento
        df = pd.read_csv(input_file[0])

        y_labels = df[select]
        x_matrix = joblib.load(input_file[1])

        # realizar un mapeo de y_labels a conjunto de categorías reducido y, posteriormente a datos numéricos
        y_labels = utiles.minimalist(y_labels)
        y_labels = utiles.mapeo_a_numeros(y_labels)

        # realizar predicciones y evaluar resultados
        modelo = classifier.entrenar_svm(x_matrix, y_labels, output_file)

        # guardar modelo
        joblib.dump(modelo, output_file)
        print("[*] Tarea finalizada")

    elif command == "clustering":
        print("[*] Tarea de clustering en marcha...")

        # obtener los datos de entrenamiento
        X_matrix = joblib.load(input_file)

        # procesar si X_matrix es o no una matriz dispersa
        if issparse(X_matrix):
            matriz_np = X_matrix.toarray()
        else:
            matriz_np = np.array(X_matrix)

        asig_clusters = clustering.kmeans(X_matrix, int(parameters))

        # obtener los valores máximos y minimos de los vectores para escalar los valores de pertenencia
        valor_maximo = np.max(matriz_np)
        valor_minimo = np.min(matriz_np)

        # crear K columnas y llenarlas en función de pertenencia al cluster
        valores_unicos = list(set(asig_clusters))
        cont_valores_unicos = len(valores_unicos)

        nueva_columna = np.array([], dtype=np.float32)
        for valor in valores_unicos:
            for instancia in asig_clusters:
                if instancia == valor:
                    nueva_columna = np.append(nueva_columna, valor_maximo)
                else:
                    nueva_columna = np.append(nueva_columna, valor_minimo)

            # añadir una columna por cada cluster
            matriz_np = np.column_stack((matriz_np, nueva_columna))
            nueva_columna = np.array([], dtype=np.float32)

        # guardar la estructura matricial en el directorio de trabajo
        joblib.dump(matriz_np, output_file)

        print("[*] Tarea finalizada")


    elif command == "Q2":
        print("[*] Obteniendo resultados para Research Question 2...")

        parameters = parameters.split(",")
        kmin = int(parameters[0])
        kmax = int(parameters[1])

        # configuración inicial del gráfico
        ancho_barra = 0.35

        # iterar sobre los distintos tipos de preprocesado
        input_files = input_file.split(",")
        for file in input_files:
            # calcular fscore del dataset original
            df = pd.read_csv("train.csv")
            y_labels = df[select]
            x_matrix = joblib.load(file)

            y_labels = utiles.minimalist(y_labels)
            y_labels = utiles.mapeo_a_numeros(y_labels)

            # realizar predicciones y obtener fscore
            _, _, fscore_sin = classifier.devolver_fscore_svm(x_matrix, y_labels)

            # calcular fscore del dataset modificado
            if issparse(x_matrix):
                matriz_np = x_matrix.toarray()
            else:
                matriz_np = np.array(x_matrix)

            asig_clusters = clustering.buscarCodo(x_matrix, kmin, kmax)

            # obtener los valores máximos y minimos de los vectores para escalar los valores de pertenencia
            valor_maximo = np.max(matriz_np)
            valor_minimo = np.min(matriz_np)

            # crear K columnas y llenarlas en función de pertenencia al cluster
            valores_unicos = list(set(asig_clusters))
            cont_valores_unicos = len(valores_unicos)

            nueva_columna = np.array([], dtype=np.float32)
            for valor in valores_unicos:
                for instancia in asig_clusters:
                    if instancia == valor:
                        nueva_columna = np.append(nueva_columna, valor_maximo)
                    else:
                        nueva_columna = np.append(nueva_columna, valor_minimo)

                # añadir una columna por cada cluster
                matriz_np = np.column_stack((matriz_np, nueva_columna))
                nueva_columna = np.array([], dtype=np.float32)
            
            # obtener fscore de nuevo formato matricial
            _, _, fscore_con = classifier.devolver_fscore_svm(matriz_np, y_labels)

            # configurar gráfico
            plt.bar("HOLA", fscore_sin, width=ancho_barra, color='orange', label='Naranja: sin clustering')
            plt.bar("ADIOS" + ancho_barra, fscore_con, width=ancho_barra, color='blue', label='Azul: con clustering')

        plt.show()
        print("[*] Resultados dibujados")

    elif command == "doQ1":

        if not os.path.exists(input_file):
            print(f"[!]Error: El archivo {input_file} no existe.")
            exit(0)

        precisions = []
        recalls = []
        fscores = []

        # cargar el CSV
        df = pd.read_csv(input_file)

        y_labels = df["gs_text34"]
        y_labels = utiles.minimalist(y_labels)
        y_labels = utiles.mapeo_a_numeros(y_labels)
        documentos_crudos = df["open_response"]
        documentos_preprocesados = preprocessor.preprocesar_texto(documentos_crudos)
        tecnicas_vectorizacion = ["bow", "tfidf", "d2v", "transformers"]


        for i in tecnicas_vectorizacion:
            if i == "bow":
                x_matrix = preprocessor.bow(documentos_preprocesados)
            elif i == "tfidf":
                x_matrix = preprocessor.tfidf(documentos_preprocesados)
            elif i == "d2v":
                x_matrix = preprocessor.we(documentos_preprocesados)
            else:
                x_matrix = preprocessor.transformers(documentos_preprocesados)

            precision, recall, fscore = classifier.devolver_fscore_svm(x_matrix, y_labels)
            precisions.append(precision)
            recalls.append(recall)
            fscores.append(fscore)

        # Configuración del gráfico
        bar_width = 0.2
        index = np.arange(len(tecnicas_vectorizacion))

        # Gráfico de barras para la precisión
        plt.subplot(3, 1, 1)
        plt.bar(index, precisions, bar_width, color='b', label='Precision')
        plt.xlabel('Técnicas de Vectorización')
        plt.ylabel('Precision')
        plt.title('Comparación de Precision para Técnicas de Vectorización')
        plt.xticks(index, tecnicas_vectorizacion)
        plt.legend()

        # Gráfico de barras para el recall
        plt.subplot(3, 1, 2)
        plt.bar(index, recalls, bar_width, color='g', label='Recall')
        plt.xlabel('Técnicas de Vectorización')
        plt.ylabel('Recall')
        plt.title('Comparación de Recall para Técnicas de Vectorización')
        plt.xticks(index, tecnicas_vectorizacion)
        plt.legend()

        # Gráfico de barras para el F-score
        plt.subplot(3, 1, 3)
        plt.bar(index, fscores, bar_width, color='r', label='F-score')
        plt.xlabel('Técnicas de Vectorización')
        plt.ylabel('F-score')
        plt.title('Comparación de F-score para Técnicas de Vectorización')
        plt.xticks(index, tecnicas_vectorizacion)
        plt.legend()

        # Ajustar el diseño para evitar superposiciones
        plt.tight_layout()

        # Mostrar el gráfico
        plt.show()


    else:
        print("[!]Error: el parámetro -d no es válido. Debe ser 'preprocess', 'classify' o 'clustering'")
        exit(0)
import os
import sys
import getopt
import joblib
import utiles
import classifier
import preprocessor
import pandas as pd



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
    
    if command == "classify":
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
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def entrenar_svm(x_matrix, y_labels):
    '''
    Precondición:
        x_matrix es un dataframe de pandas que contiene estructuras vectoriales que hacen referencia a las
        autopsias verbales de cada instancia. 
        y_labels es una lista con las etiquetas reales de las instancias.
        el orden de las instancias en x_matrix es el mismo que en y_labels.
    Poscondición:
        imprime figuras de mérito para una tarea de clasificación. Concretamente la precisión, el recall y el fscore.
        devuelve un modelo SVM entrenado para las instancias de x_matrix e y_labels.
    '''
    # separar train y test (70%-30%)
    x_train, x_test, y_train, y_test = train_test_split(x_matrix, y_labels, test_size=0.3, random_state=42)

    # entrenar un modelo SVM
    clf = svm.SVC(kernel='linear')
    clf.fit(x_train, y_train)

    # evaluar modelo creado
    pred = clf.predict(x_test)

    precision = precision_score(y_test, pred, average='weighted')
    recall = recall_score(y_test, pred, average='weighted')
    fscore = f1_score(y_test, pred, average='weighted')
    print(f"Precisión del modelo: {precision}")
    print(f"Recall del modelo: {recall}")
    print(f"FScore del modelo: {fscore}")


    return clf


def devolver_fscore_svm(x_matrix, y_labels, v):
    '''
    Precondición:
        x_matrix es un dataframe de pandas que contiene estructuras vectoriales que hacen referencia a las
        autopsias verbales de cada instancia. 
        y_labels es una lista con las etiquetas reales de las instancias.
        el orden de las instancias en x_matrix es el mismo que en y_labels.
        v es la técnica de vectorización utilizada sobre los datos.
    Poscondición:
        devuelve la precisión, el recall y el fscore para una tarea de clasificación con
        el algoritmo SVM aplicado a x_matrix e y_labels, y guarda una imagen de la matriz de confusión del modelo obtenido.
    '''
    # separar train y test (70%-30%)
    x_train, x_test, y_train, y_test = train_test_split(x_matrix, y_labels, test_size=0.3, random_state=42)

    # entrenar un modelo SVM
    clf = svm.SVC(kernel='linear')
    clf.fit(x_train, y_train)

    # evaluar modelo creado
    pred = clf.predict(x_test)

    precision = precision_score(y_test, pred, average='weighted')
    recall = recall_score(y_test, pred, average='weighted')
    fscore = f1_score(y_test, pred, average='weighted')

    # Calcular la matriz de confusión
    cm = confusion_matrix(y_test, pred)

    # Visualizar la matriz de confusión con seaborn
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Matriz de Confusión")
    plt.xlabel("Predicciones")
    plt.ylabel("Valores verdaderos")
    plt.savefig('matriz_confusion' + v + '.png')

    return precision, recall, fscore
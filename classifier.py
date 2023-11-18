from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score


def entrenar_svm(x_matrix, y_labels, output_file):
    '''
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
import nltk
import torch
import string
import pandas as pd
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import AutoTokenizer, AutoModel
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')  # Descarga la lista de stop words en inglés
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')


def normalizarTexto(texto):
    '''
    Precondiciones:
        texto es una cadena de caracteres.
    Postcondiciones:
        La función convierte el texto a minúsculas y devuelve el texto normalizado.
    '''
    return texto.lower()


def eliminarSignosPuntuacion(texto):
    '''
    Precondiciones:
        texto es una cadena de caracteres.
    Postcondiciones:
        La función elimina los signos de puntuación del texto y devuelve el texto sin puntuación (con los caracteres alfabéticos).
    '''
    # Obtiene la lista de puntuaciones oficial del módulo string
    puntuacion = string.punctuation

    # Crea un traductor para eemplazar todo lo que haya en la lista por ''
    translator = str.maketrans('', '', puntuacion)

    # Traducir / Reemplazar
    texto_sin_puntuacion = texto.translate(translator)

    return texto_sin_puntuacion


def eliminarStopWords(texto):
    '''
    Precondiciones:
        texto es una cadena de caracteres.
    Postcondiciones:
        La función elimina las palabras consideradas como stop words del texto y devuelve el texto sin stop words.
    '''
    # Obtiene la lista de stop words en inglés
    stop_words = set(stopwords.words("english"))

    # Divide el texto en palabras
    palabras = texto.split()

    # Filtra las palabras que no son stop words
    palabras_filtradas = [palabra for palabra in palabras if palabra.lower() not in stop_words]

    # Re-construye el texto sin las stop words
    texto_sin_stop_words = " ".join(palabras_filtradas)
    
    return texto_sin_stop_words


def get_wordnet_pos(word):
    '''
    Función auxiliar para obtener la categoría gramatical de las palabras (necesaria para la lematización)
    Precondiciones:
        word es una cadena de caracteres que representa una palabra.
    Postcondiciones:
        La función devuelve la categoría gramatical de la palabra en el formato utilizado por WordNet (ADJ, NOUN, VERB, ADV).
    '''

    tag = nltk.pos_tag([word])[0][1][0].upper()


    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def lematizar(texto):
    '''
    Precondiciones:
        texto es una cadena de caracteres.
    Postcondiciones:
        La función lematiza las palabras en el texto, utilizando la categoría gramatical apropiada, y devuelve el texto lematizado.
    '''
    texto = nltk.word_tokenize(texto)
    
    # Inicializar el lematizador
    lemmatizer = WordNetLemmatizer()

    # Lematizar cada palabra y agregarla a una lista
    palabras_lematizadas = []
    for palabra in texto:
        pos = get_wordnet_pos(palabra)
        palabra_l = lemmatizer.lemmatize(palabra, pos=pos)
        palabras_lematizadas.append(palabra_l)
    
    # Unir las palabras lematizadas en un solo string y devolverlo
    texto_lematizado = ' '.join(palabras_lematizadas)
    return texto_lematizado


def eliminarSiNoInfo(texto, n_palabras):
    '''
    Precondiciones:
        texto es una cadena de caracteres.
        n_palabras es un entero no negativo que representa el número mínimo de palabras requeridas en el texto.
    Postcondiciones:
        Si el número de palabras en el texto es menor que n_palabras, la función devuelve None (elimina el texto).
        Si el número de palabras en el texto es mayor o igual que n_palabras, la función devuelve el texto sin cambios.
    '''
    palabras = texto.split()
    num_palabras = len(palabras)
    if num_palabras < n_palabras:
        # Si tiene menos de n_palabras, elimina el texto
        return None
    else:
        return texto


def preprocesar_texto(texto_crudo):
    '''
        Precondición:
            texto_crudo es una lista de documentos que no han recibido ningún preproceso.
        Poscondición:
            Devuelve la lista de documentos recibida como parámetro sin signos de puntuación,
            sin mayúsculas, sin stopwords y con las palabras en su forma raíz.
            Ningún documento que estuviese formado por sólo 3 palabras se tendrá en cuenta.
    '''
    listaFilas = []
    for i, texto in texto_crudo.items():
        texto = eliminarSiNoInfo(texto, n_palabras=3)
        texto = eliminarSignosPuntuacion(texto)
        texto = normalizarTexto(texto)
        texto = eliminarStopWords(texto)
        texto = lematizar(texto)
        if texto is not None:       # Puede ser None si se ha eliminado por no tener información relevante
            listaFilas.append(texto)
    columnaProcesada = pd.Series(listaFilas)

    return columnaProcesada


def bow(documentos):
    '''
    Precondición:
        documentos es una lista de strings.
    Poscondición:
        Devuelve la representación vectorial Bag Of Words para documentos en forma de matriz.
    '''
    vectorizer = CountVectorizer()
    bagofwords = vectorizer.fit_transform(documentos)
    matriz_bow = bagofwords.toarray()
    
    return matriz_bow


def tfidf(documentos):
    '''
    Precondición:
        documentos es una lista de strings.
    Poscondición:
        Devuelve la representación vectorial tf-idf para documentos en forma de matriz.
    '''
    # Crear un vectorizador TF-IDF
    vectorizador = TfidfVectorizer()

    # Ajustar el vectorizador a los documentos recibidos como parámetro
    matriz_tfidf = vectorizador.fit_transform(documentos)
    print(matriz_tfidf.shape)

    return matriz_tfidf


def we(documentos):
    '''
    Precondición:
        documentos es una lista de strings.
    Poscondición:
        Devuelve el word embedding de los documentos, entrenados por un modelo doc2vec.
    '''
    tokenized_data = [doc.split() for doc in documentos]
    tagged_data = [TaggedDocument(words=words, tags=[str(idx)]) for idx, words in enumerate(tokenized_data)]

    # Crear y entrenar el modelo Doc2Vec
    model = Doc2Vec(vector_size=100, min_count=1, epochs=100, workers=2)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=model.epochs)

    # Obtener los vectores de los documentos
    vectores_documentos = [model.infer_vector(words) for words in tokenized_data]
    print(len(vectores_documentos[5]))

    return vectores_documentos


def transformers(documentos):
    '''
    Precondición:
        documentos es una lista de strings.
    Poscondición:
        La función devuelve una lista de representaciones numéricas de los documentos proporcionados.
        Cada elemento de la lista es un array NumPy que representa la representación del 
        documento correspondiente.
    '''
    # Cargar el modelo preentrenado y el tokenizador
    modelo_nombre = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(modelo_nombre)
    modelo = AutoModel.from_pretrained(modelo_nombre)
    # Tokenizar y obtener representación para cada documento
    representaciones = []
    i = 0
    for documento in documentos:
        # Tokenizar el documento
        tokens = tokenizer(documento, return_tensors="pt")

        # Obtener la representación del modelo
        with torch.no_grad():
            salida = modelo(**tokens)

        # Puedes tomar la salida de la capa de pooling para obtener una representación del documento
        representacion_documento = torch.mean(salida.last_hidden_state, dim=1).squeeze().numpy()
        representaciones.append(representacion_documento)
        i += 1

    print(len(representaciones[5]))

    return representaciones
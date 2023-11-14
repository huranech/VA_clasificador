import string
import nltk
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
nltk.download('stopwords')  # Descarga la lista de stop words en inglés


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


def tfidf(documentos):
    '''
    '''
    # Crear un vectorizador TF-IDF
    vectorizador = TfidfVectorizer()

    # Ajustar el vectorizador a los documentos recibidos como parámetro
    matriz_tfidf = vectorizador.fit_transform(documentos)

    return matriz_tfidf


def we(documentos):
    '''
    '''
    pass


def transformers(documentos):
    '''
    '''
    pass


def obtenerWE(documents, vector_size=1000, window=5, min_count=1):
    '''
    Este método devuelve la estructura de datos de Word Embedding
    Precondiciones:
        documents es una lista de documentos, donde cada documento es una lista de palabras.
        vector_size es un entero que representa la dimensión del espacio vectorial de embedding (por defecto, 1000).
        window es un entero que representa la ventana de contexto para el modelo Doc2Vec (por defecto, 5).
        min_count es un entero que representa el número mínimo de ocurrencias de una palabra (por defecto, 1).
    Postcondiciones:
        La función utiliza el modelo Doc2Vec para generar representaciones vectoriales de los documentos.
        Devuelve una matriz NumPy que contiene las representaciones vectoriales de los documentos.
    '''
    # Etiqueta los documentos
    tagged_data = [TaggedDocument(words=words, tags=[str(i)]) for i, words in enumerate(documents)]

    modelo = Doc2Vec(vector_size=400, window=2, min_count=1, workers=4, epochs=100)

    # Construye el vocabulario
    modelo.build_vocab(tagged_data)

    # Obtiene las representaciones vectoriales de todos los documentos
    vectors = [modelo.dv[str(i)] for i in range(len(documents))]

    # Convierte la lista de vectores en una matriz NumPy
    X = np.array(vectors)
    
    return X
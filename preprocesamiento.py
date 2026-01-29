"""
Módulo de Preprocesamiento de Texto
====================================

Este módulo contiene todas las funciones necesarias para preprocesar
los mensajes de correo antes de ser utilizados por el modelo de Naive Bayes.

El preprocesamiento incluye:
1. Normalización (minúsculas y eliminación de puntuación)
2. Tokenización (división en palabras)
3. Eliminación de stopwords (palabras comunes sin significado)
4. Conversión a Bag of Words (representación numérica)
"""

import re
import string
from collections import Counter


# Lista de stopwords en español - palabras comunes que no aportan información útil
# para la clasificación de spam/ham
STOPWORDS_ESPANOL = {
    'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'ser', 'se',
    'no', 'haber', 'por', 'con', 'su', 'para', 'como', 'estar',
    'tener', 'le', 'lo', 'todo', 'pero', 'más', 'hacer', 'o',
    'poder', 'decir', 'este', 'ir', 'otro', 'ese', 'la', 'si',
    'me', 'ya', 'ver', 'porque', 'dar', 'cuando', 'él', 'muy',
    'sin', 'vez', 'mucho', 'saber', 'qué', 'sobre', 'mi', 'alguno',
    'mismo', 'yo', 'también', 'hasta', 'año', 'dos', 'querer',
    'entre', 'así', 'primero', 'desde', 'grande', 'eso', 'ni',
    'nos', 'llegar', 'pasar', 'tiempo', 'ella', 'sí', 'día',
    'uno', 'bien', 'poco', 'deber', 'entonces', 'poner', 'cosa',
    'tanto', 'hombre', 'parecer', 'nuestro', 'tan', 'donde',
    'ahora', 'parte', 'después', 'vida', 'quedar', 'siempre',
    'creer', 'hablar', 'llevar', 'dejar', 'nada', 'cada', 'seguir',
    'menos', 'nuevo', 'encontrar', 'algo', 'solo', 'país', 'menos',
    'mientras', 'mujer', 'aquel', 'así', 'leer', 'mundo', 'aunque',
    'trabajar', 'problema', 'semana', 'hacer', 'empezar', 'mirar',
    'casa', 'cambiar', 'señor', 'dar', 'sistema', 'tener', 'hecho',
    'trabajo', 'niño', 'estado', 'todavía', 'otro', 'decir', 'poder',
    'querer', 'saber', 'ir', 'decir', 'haber', 'ser', 'estar'
}


def normalizar_texto(texto):
    """
    Normaliza un texto convirtiéndolo a minúsculas y eliminando signos de puntuación.
    
    Esta función es el primer paso del preprocesamiento. Convierte todo el texto
    a minúsculas para que "Dinero" y "dinero" sean tratadas como la misma palabra,
    y elimina signos de puntuación que no aportan información para la clasificación.
    
    Parámetros:
    -----------
    texto : str
        El texto original que se desea normalizar
        
    Retorna:
    --------
    str
        El texto normalizado en minúsculas y sin puntuación
        
    Ejemplo:
    --------
    >>> normalizar_texto("¡Gana dinero RÁPIDO!")
    'gana dinero rápido'
    """
    if not isinstance(texto, str):
        texto = str(texto)
    
    # Convertir a minúsculas
    texto = texto.lower()
    
    # Eliminar signos de puntuación usando expresiones regulares
    # [^\w\s] significa: cualquier carácter que NO sea alfanumérico o espacio
    texto = re.sub(r'[^\w\s]', ' ', texto)
    
    # Reemplazar múltiples espacios por un solo espacio
    texto = re.sub(r'\s+', ' ', texto)
    
    # Eliminar espacios al inicio y final
    texto = texto.strip()
    
    return texto


def tokenizar(texto):
    """
    Divide un texto en tokens (palabras individuales).
    
    La tokenización es el proceso de dividir un texto en unidades más pequeñas,
    generalmente palabras. Esta función toma un texto normalizado y lo divide
    en una lista de palabras individuales.
    
    Parámetros:
    -----------
    texto : str
        El texto normalizado que se desea tokenizar
        
    Retorna:
    --------
    list
        Lista de tokens (palabras) del texto
        
    Ejemplo:
    --------
    >>> tokenizar("gana dinero rápido")
    ['gana', 'dinero', 'rápido']
    """
    if not texto:
        return []
    
    # Dividir el texto por espacios en blanco
    tokens = texto.split()
    
    # Filtrar tokens vacíos (por si acaso)
    tokens = [token for token in tokens if token.strip()]
    
    return tokens


def eliminar_stopwords(tokens):
    """
    Elimina las stopwords (palabras comunes) de una lista de tokens.
    
    Las stopwords son palabras muy frecuentes que generalmente no aportan
    información útil para distinguir entre spam y ham. Por ejemplo, palabras
    como "el", "la", "de", "y" aparecen en ambos tipos de correos y no
    ayudan a la clasificación.
    
    Eliminar stopwords:
    - Reduce el tamaño del vocabulario
    - Mejora la eficiencia del modelo
    - Enfoca el aprendizaje en palabras más significativas
    
    Parámetros:
    -----------
    tokens : list
        Lista de tokens (palabras) de las que se eliminarán stopwords
        
    Retorna:
    --------
    list
        Lista de tokens sin stopwords
        
    Ejemplo:
    --------
    >>> eliminar_stopwords(['el', 'dinero', 'es', 'fácil'])
    ['dinero', 'fácil']
    """
    if not tokens:
        return []
    
    # Filtrar tokens: mantener solo los que NO están en STOPWORDS_ESPANOL
    tokens_filtrados = [token for token in tokens if token not in STOPWORDS_ESPANOL]
    
    return tokens_filtrados


def crear_bag_of_words(mensajes):
    """
    Crea una representación Bag of Words (Bolsa de Palabras) de los mensajes.
    
    Bag of Words es una técnica que convierte textos en vectores numéricos.
    Cada mensaje se representa como un vector donde cada posición corresponde
    a una palabra del vocabulario, y el valor indica cuántas veces aparece
    esa palabra en el mensaje.
    
    Ejemplo:
    --------
    Vocabulario: ['dinero', 'fácil', 'reunión']
    Mensaje: "dinero fácil dinero" -> [2, 1, 0]
             (dinero aparece 2 veces, fácil 1 vez, reunión 0 veces)
    
    Parámetros:
    -----------
    mensajes : list
        Lista de mensajes preprocesados (cada uno es una lista de tokens)
        
    Retorna:
    --------
    tuple: (vocabulario, vectores)
        - vocabulario: diccionario que mapea cada palabra a un índice único
        - vectores: lista de vectores (listas) donde cada vector representa un mensaje
        
    Ejemplo:
    --------
    >>> mensajes = [['dinero', 'fácil'], ['reunión', 'mañana']]
    >>> vocab, vectores = crear_bag_of_words(mensajes)
    >>> vocab
    {'dinero': 0, 'fácil': 1, 'reunión': 2, 'mañana': 3}
    >>> vectores
    [[1, 1, 0, 0], [0, 0, 1, 1]]
    """
    # Paso 1: Crear el vocabulario (conjunto único de todas las palabras)
    # Usamos un conjunto (set) para obtener palabras únicas
    todas_las_palabras = []
    for mensaje in mensajes:
        todas_las_palabras.extend(mensaje)
    
    # Obtener palabras únicas y ordenarlas para consistencia
    vocabulario_palabras = sorted(set(todas_las_palabras))
    
    # Crear un diccionario que mapea cada palabra a un índice único
    # Esto permite convertir palabras en números de forma eficiente
    vocabulario = {palabra: indice for indice, palabra in enumerate(vocabulario_palabras)}
    
    # Paso 2: Crear los vectores para cada mensaje
    vectores = []
    for mensaje in mensajes:
        # Inicializar un vector de ceros con el tamaño del vocabulario
        vector = [0] * len(vocabulario)
        
        # Contar la frecuencia de cada palabra en el mensaje
        # Usamos Counter para contar palabras eficientemente
        contador = Counter(mensaje)
        
        # Para cada palabra en el mensaje, actualizar su posición en el vector
        for palabra, frecuencia in contador.items():
            if palabra in vocabulario:
                indice = vocabulario[palabra]
                vector[indice] = frecuencia
        
        vectores.append(vector)
    
    return vocabulario, vectores


def preprocesar_mensaje(texto):
    """
    Preprocesa un solo mensaje aplicando todas las transformaciones.
    
    Esta función aplica el pipeline completo de preprocesamiento a un mensaje:
    1. Normalización
    2. Tokenización
    3. Eliminación de stopwords
    
    Es útil para preprocesar mensajes individuales antes de hacer predicciones.
    
    Parámetros:
    -----------
    texto : str
        El mensaje original que se desea preprocesar
        
    Retorna:
    --------
    list
        Lista de tokens preprocesados (palabras relevantes)
        
    Ejemplo:
    --------
    >>> preprocesar_mensaje("¡Gana dinero RÁPIDO haciendo clic!")
    ['gana', 'dinero', 'rápido', 'haciendo', 'clic']
    """
    # Aplicar el pipeline completo
    texto_normalizado = normalizar_texto(texto)
    tokens = tokenizar(texto_normalizado)
    tokens_sin_stopwords = eliminar_stopwords(tokens)
    
    return tokens_sin_stopwords


def preprocesar_dataset(datos):
    """
    Preprocesa todo un dataset de mensajes.
    
    Esta función toma un conjunto de datos (lista de textos) y aplica
    el preprocesamiento completo a cada uno, retornando:
    - Los mensajes preprocesados (listas de tokens)
    - El vocabulario completo
    - Los vectores Bag of Words
    
    Parámetros:
    -----------
    datos : list
        Lista de mensajes (strings) que se desean preprocesar
        
    Retorna:
    --------
    tuple: (mensajes_preprocesados, vocabulario, vectores)
        - mensajes_preprocesados: lista de listas de tokens
        - vocabulario: diccionario palabra -> índice
        - vectores: lista de vectores numéricos (Bag of Words)
        
    Ejemplo:
    --------
    >>> mensajes = ["Gana dinero rápido", "Reunión mañana"]
    >>> preproc, vocab, vecs = preprocesar_dataset(mensajes)
    >>> preproc
    [['gana', 'dinero', 'rápido'], ['reunión', 'mañana']]
    """
    # Preprocesar cada mensaje individualmente
    mensajes_preprocesados = []
    for mensaje in datos:
        mensaje_preprocesado = preprocesar_mensaje(mensaje)
        mensajes_preprocesados.append(mensaje_preprocesado)
    
    # Crear el Bag of Words con todos los mensajes preprocesados
    vocabulario, vectores = crear_bag_of_words(mensajes_preprocesados)
    
    return mensajes_preprocesados, vocabulario, vectores


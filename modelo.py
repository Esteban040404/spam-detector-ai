"""
Módulo del Modelo Naive Bayes
==============================

Este módulo implementa el clasificador Naive Bayes desde cero para
detectar spam en mensajes de correo electrónico.

El algoritmo Naive Bayes está basado en el Teorema de Bayes y asume
que las características (palabras) son independientes entre sí.
"""

import math
import pickle
import os
from collections import Counter, defaultdict
from preprocesamiento import preprocesar_mensaje


class NaiveBayesSpamDetector:
    """
    Clasificador Naive Bayes para detectar spam en mensajes de correo.
    
    Este modelo utiliza el Teorema de Bayes para calcular la probabilidad
    de que un mensaje sea spam o ham (no spam) basándose en las palabras
    que contiene.
    
    El modelo funciona en dos fases:
    1. Entrenamiento: aprende las probabilidades de palabras en cada clase
    2. Predicción: calcula la probabilidad de cada clase para nuevos mensajes
    """
    
    def __init__(self, alpha=1.0):
        """
        Inicializa el clasificador Naive Bayes.
        
        Parámetros:
        -----------
        alpha : float, default=1.0
            Parámetro de suavizado de Laplace. Previene problemas cuando
            una palabra no aparece en el entrenamiento de una clase.
            alpha=1.0 es el suavizado estándar (add-one smoothing).
            
            Fórmula del suavizado:
            P(palabra|clase) = (frecuencia + alpha) / (total_palabras + alpha * vocabulario)
        """
        self.alpha = alpha
        
        # Probabilidades a priori: P(spam) y P(ham)
        # Se calcularán durante el entrenamiento
        self.prob_spam = 0.0
        self.prob_ham = 0.0
        
        # Contadores de palabras por clase
        # spam_words: cuenta cuántas veces aparece cada palabra en mensajes spam
        # ham_words: cuenta cuántas veces aparece cada palabra en mensajes ham
        self.spam_words = Counter()
        self.ham_words = Counter()
        
        # Totales de palabras en cada clase
        self.total_spam_words = 0
        self.total_ham_words = 0
        
        # Vocabulario: conjunto de todas las palabras únicas vistas durante el entrenamiento
        self.vocabulario = set()
        
        # Probabilidades condicionales logarítmicas (para evitar underflow)
        # Se calculan durante el entrenamiento
        self.log_prob_spam_words = {}
        self.log_prob_ham_words = {}
        
        # Flag para saber si el modelo ha sido entrenado
        self.entrenado = False
    
    def entrenar(self, X_train, y_train):
        """
        Entrena el modelo Naive Bayes con los datos de entrenamiento.
        
        Durante el entrenamiento, el modelo:
        1. Calcula las probabilidades a priori P(spam) y P(ham)
        2. Cuenta las frecuencias de palabras en cada clase
        3. Calcula las probabilidades condicionales P(palabra|clase)
        4. Aplica suavizado de Laplace para manejar palabras nuevas
        
        Parámetros:
        -----------
        X_train : list
            Lista de mensajes preprocesados (cada mensaje es una lista de tokens/palabras)
        y_train : list
            Lista de etiquetas correspondientes ('spam' o 'ham')
            
        Ejemplo:
        --------
        >>> modelo = NaiveBayesSpamDetector()
        >>> mensajes = [['dinero', 'fácil'], ['reunión', 'mañana']]
        >>> etiquetas = ['spam', 'ham']
        >>> modelo.entrenar(mensajes, etiquetas)
        """
        if len(X_train) != len(y_train):
            raise ValueError("X_train y y_train deben tener la misma longitud")
        
        # Paso 1: Calcular probabilidades a priori P(spam) y P(ham)
        # Estas son simplemente la proporción de mensajes de cada clase
        total_mensajes = len(y_train)
        spam_count = sum(1 for etiqueta in y_train if etiqueta == 'spam')
        ham_count = total_mensajes - spam_count
        
        # P(spam) = número de mensajes spam / número total de mensajes
        self.prob_spam = spam_count / total_mensajes
        # P(ham) = número de mensajes ham / número total de mensajes
        self.prob_ham = ham_count / total_mensajes
        
        # Paso 2: Contar frecuencias de palabras en cada clase
        # Iteramos sobre cada mensaje y su etiqueta
        for mensaje, etiqueta in zip(X_train, y_train):
            if etiqueta == 'spam':
                # Si es spam, agregamos las palabras al contador de spam
                self.spam_words.update(mensaje)
                self.total_spam_words += len(mensaje)
            else:  # ham
                # Si es ham, agregamos las palabras al contador de ham
                self.ham_words.update(mensaje)
                self.total_ham_words += len(mensaje)
            
            # Agregar todas las palabras al vocabulario
            self.vocabulario.update(mensaje)
        
        # Paso 3: Calcular probabilidades condicionales con suavizado de Laplace
        # Usamos logaritmos para evitar problemas numéricos (underflow)
        # cuando multiplicamos muchas probabilidades pequeñas
        
        tamaño_vocabulario = len(self.vocabulario)
        
        # Calcular log-probabilidades para palabras en spam
        # Fórmula: log(P(palabra|spam)) = log((frecuencia + alpha) / (total + alpha * V))
        for palabra in self.vocabulario:
            frecuencia_spam = self.spam_words.get(palabra, 0)
            # Suavizado de Laplace aplicado
            prob = (frecuencia_spam + self.alpha) / (self.total_spam_words + self.alpha * tamaño_vocabulario)
            # Usar logaritmo para evitar underflow en la multiplicación posterior
            self.log_prob_spam_words[palabra] = math.log(prob)
        
        # Calcular log-probabilidades para palabras en ham
        for palabra in self.vocabulario:
            frecuencia_ham = self.ham_words.get(palabra, 0)
            # Suavizado de Laplace aplicado
            prob = (frecuencia_ham + self.alpha) / (self.total_ham_words + self.alpha * tamaño_vocabulario)
            # Usar logaritmo para evitar underflow
            self.log_prob_ham_words[palabra] = math.log(prob)
        
        # También guardar las probabilidades logarítmicas a priori
        self.log_prob_spam = math.log(self.prob_spam) if self.prob_spam > 0 else float('-inf')
        self.log_prob_ham = math.log(self.prob_ham) if self.prob_ham > 0 else float('-inf')
        
        self.entrenado = True
    
    def _calcular_log_probabilidad(self, mensaje, clase):
        """
        Calcula la probabilidad logarítmica de que un mensaje pertenezca a una clase.
        
        Esta es una función auxiliar interna que implementa el Teorema de Bayes:
        
        P(clase|mensaje) ∝ P(clase) * ∏ P(palabra|clase)
        
        Usamos logaritmos para convertir la multiplicación en suma (más estable numéricamente):
        
        log(P(clase|mensaje)) = log(P(clase)) + Σ log(P(palabra|clase))
        
        Parámetros:
        -----------
        mensaje : list
            Lista de tokens (palabras) del mensaje
        clase : str
            Clase para la cual calcular la probabilidad ('spam' o 'ham')
            
        Retorna:
        --------
        float
            Log-probabilidad de que el mensaje pertenezca a la clase
        """
        if clase == 'spam':
            # Empezar con log(P(spam))
            log_prob = self.log_prob_spam
            prob_palabras = self.log_prob_spam_words
        else:  # ham
            # Empezar con log(P(ham))
            log_prob = self.log_prob_ham
            prob_palabras = self.log_prob_ham_words
        
        # Sumar log(P(palabra|clase)) para cada palabra en el mensaje
        # Esto es equivalente a multiplicar P(palabra|clase) para todas las palabras
        for palabra in mensaje:
            if palabra in prob_palabras:
                # Si la palabra está en el vocabulario, usar su probabilidad conocida
                log_prob += prob_palabras[palabra]
            else:
                # Si la palabra no está en el vocabulario (Out of Vocabulary - OOV)
                # Usamos el suavizado: P(OOV|clase) = alpha / (total + alpha * V)
                if clase == 'spam':
                    tamaño_vocab = len(self.vocabulario)
                    prob_oov = self.alpha / (self.total_spam_words + self.alpha * tamaño_vocab)
                else:
                    tamaño_vocab = len(self.vocabulario)
                    prob_oov = self.alpha / (self.total_ham_words + self.alpha * tamaño_vocab)
                log_prob += math.log(prob_oov)
        
        return log_prob
    
    def predecir_proba(self, mensaje):
        """
        Calcula las probabilidades de que un mensaje sea spam o ham.
        
        Usa el Teorema de Bayes para calcular ambas probabilidades y las
        normaliza para que sumen 1.0.
        
        Parámetros:
        -----------
        mensaje : str o list
            El mensaje a clasificar. Puede ser un string (se preprocesará)
            o una lista de tokens ya preprocesados
            
        Retorna:
        --------
        dict
            Diccionario con las probabilidades:
            {'spam': probabilidad_spam, 'ham': probabilidad_ham}
        """
        if not self.entrenado:
            raise ValueError("El modelo debe ser entrenado antes de hacer predicciones")
        
        # Si el mensaje es un string, preprocesarlo primero
        if isinstance(mensaje, str):
            mensaje_tokens = preprocesar_mensaje(mensaje)
        else:
            mensaje_tokens = mensaje
        
        # Calcular log-probabilidades para ambas clases
        log_prob_spam = self._calcular_log_probabilidad(mensaje_tokens, 'spam')
        log_prob_ham = self._calcular_log_probabilidad(mensaje_tokens, 'ham')
        
        # Para normalizar las probabilidades, necesitamos convertir de log a probabilidad
        # y luego normalizar. Pero para evitar overflow, usamos una técnica especial:
        # Restamos el máximo de las dos log-probabilidades antes de exponenciar
        
        max_log_prob = max(log_prob_spam, log_prob_ham)
        log_prob_spam_shifted = log_prob_spam - max_log_prob
        log_prob_ham_shifted = log_prob_ham - max_log_prob
        
        # Convertir a probabilidades reales (exponenciar)
        prob_spam = math.exp(log_prob_spam_shifted)
        prob_ham = math.exp(log_prob_ham_shifted)
        
        # Normalizar para que sumen 1.0
        suma = prob_spam + prob_ham
        if suma > 0:
            prob_spam_normalizada = prob_spam / suma
            prob_ham_normalizada = prob_ham / suma
        else:
            # Caso edge: ambas probabilidades son 0
            prob_spam_normalizada = 0.5
            prob_ham_normalizada = 0.5
        
        return {
            'spam': prob_spam_normalizada,
            'ham': prob_ham_normalizada
        }
    
    def predecir(self, mensaje):
        """
        Predice si un mensaje es spam o ham.
        
        Calcula las probabilidades para ambas clases y retorna la clase
        con mayor probabilidad.
        
        Parámetros:
        -----------
        mensaje : str o list
            El mensaje a clasificar. Puede ser un string (se preprocesará)
            o una lista de tokens ya preprocesados
            
        Retorna:
        --------
        str
            'spam' o 'ham' según la predicción del modelo
        """
        probabilidades = self.predecir_proba(mensaje)
        
        # Retornar la clase con mayor probabilidad
        if probabilidades['spam'] > probabilidades['ham']:
            return 'spam'
        else:
            return 'ham'
    
    def obtener_palabras_importantes(self, top_n=10):
        """
        Retorna las palabras más importantes para distinguir entre spam y ham.
        
        Una palabra es "importante" si tiene probabilidades muy diferentes
        entre las dos clases. Calculamos la diferencia entre las probabilidades.
        
        Parámetros:
        -----------
        top_n : int, default=10
            Número de palabras más importantes a retornar
            
        Retorna:
        --------
        dict
            Diccionario con 'spam' y 'ham', cada uno con lista de tuplas
            (palabra, diferencia_probabilidad)
        """
        if not self.entrenado:
            return {}
        
        # Calcular diferencias para palabras en el vocabulario
        diferencias_spam = []  # Palabras más probables en spam
        diferencias_ham = []   # Palabras más probables en ham
        
        for palabra in self.vocabulario:
            prob_spam = math.exp(self.log_prob_spam_words.get(palabra, float('-inf')))
            prob_ham = math.exp(self.log_prob_ham_words.get(palabra, float('-inf')))
            
            diferencia = prob_spam - prob_ham
            
            if diferencia > 0:
                # Más probable en spam
                diferencias_spam.append((palabra, diferencia))
            else:
                # Más probable en ham
                diferencias_ham.append((palabra, abs(diferencia)))
        
        # Ordenar por diferencia (mayor diferencia = más importante)
        diferencias_spam.sort(key=lambda x: x[1], reverse=True)
        diferencias_ham.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'spam': diferencias_spam[:top_n],
            'ham': diferencias_ham[:top_n]
        }
    
    def guardar(self, ruta_archivo='modelo_entrenado.pkl'):
        """
        Guarda el modelo entrenado en un archivo usando pickle.
        
        Esto permite preservar el conocimiento aprendido y reutilizarlo
        sin necesidad de reentrenar desde cero.
        
        Parámetros:
        -----------
        ruta_archivo : str, default='modelo_entrenado.pkl'
            Ruta donde guardar el modelo
            
        Ejemplo:
        --------
        >>> modelo.entrenar(X_train, y_train)
        >>> modelo.guardar('mi_modelo.pkl')
        """
        if not self.entrenado:
            raise ValueError("El modelo debe ser entrenado antes de guardarlo")
        
        # Crear directorio si no existe
        directorio = os.path.dirname(ruta_archivo)
        if directorio and not os.path.exists(directorio):
            os.makedirs(directorio, exist_ok=True)
        
        # Convertir vocabulario de set a list para poder guardarlo
        vocabulario_lista = list(self.vocabulario)
        
        # Crear diccionario con todos los datos del modelo
        datos_modelo = {
            'alpha': self.alpha,
            'prob_spam': self.prob_spam,
            'prob_ham': self.prob_ham,
            'spam_words': dict(self.spam_words),
            'ham_words': dict(self.ham_words),
            'total_spam_words': self.total_spam_words,
            'total_ham_words': self.total_ham_words,
            'vocabulario': vocabulario_lista,
            'log_prob_spam_words': self.log_prob_spam_words,
            'log_prob_ham_words': self.log_prob_ham_words,
            'log_prob_spam': self.log_prob_spam,
            'log_prob_ham': self.log_prob_ham,
            'entrenado': self.entrenado
        }
        
        with open(ruta_archivo, 'wb') as archivo:
            pickle.dump(datos_modelo, archivo)
        
        print(f"✓ Modelo guardado exitosamente en: {ruta_archivo}")
    
    @classmethod
    def cargar(cls, ruta_archivo='modelo_entrenado.pkl'):
        """
        Carga un modelo previamente entrenado desde un archivo.
        
        Parámetros:
        -----------
        ruta_archivo : str, default='modelo_entrenado.pkl'
            Ruta del archivo donde está guardado el modelo
            
        Retorna:
        --------
        NaiveBayesSpamDetector
            Instancia del modelo cargada con el conocimiento previamente aprendido
            
        Ejemplo:
        --------
        >>> modelo = NaiveBayesSpamDetector.cargar('mi_modelo.pkl')
        >>> prediccion = modelo.predecir("Gana dinero rápido")
        """
        if not os.path.exists(ruta_archivo):
            raise FileNotFoundError(f"No se encontró el archivo: {ruta_archivo}")
        
        with open(ruta_archivo, 'rb') as archivo:
            datos_modelo = pickle.load(archivo)
        
        # Crear nueva instancia
        modelo = cls(alpha=datos_modelo['alpha'])
        
        # Restaurar todos los atributos
        modelo.prob_spam = datos_modelo['prob_spam']
        modelo.prob_ham = datos_modelo['prob_ham']
        modelo.spam_words = Counter(datos_modelo['spam_words'])
        modelo.ham_words = Counter(datos_modelo['ham_words'])
        modelo.total_spam_words = datos_modelo['total_spam_words']
        modelo.total_ham_words = datos_modelo['total_ham_words']
        modelo.vocabulario = set(datos_modelo['vocabulario'])
        modelo.log_prob_spam_words = datos_modelo['log_prob_spam_words']
        modelo.log_prob_ham_words = datos_modelo['log_prob_ham_words']
        modelo.log_prob_spam = datos_modelo['log_prob_spam']
        modelo.log_prob_ham = datos_modelo['log_prob_ham']
        modelo.entrenado = datos_modelo['entrenado']
        
        print(f"✓ Modelo cargado exitosamente desde: {ruta_archivo}")
        print(f"  - Vocabulario: {len(modelo.vocabulario)} palabras")
        print(f"  - P(spam) = {modelo.prob_spam:.4f}, P(ham) = {modelo.prob_ham:.4f}")
        
        return modelo
    
    def continuar_entrenamiento(self, X_nuevos, y_nuevos):
        """
        Continúa el entrenamiento del modelo con nuevos datos.
        
        Esta función permite aprendizaje incremental: el modelo puede seguir
        aprendiendo de nuevos ejemplos sin perder el conocimiento previo.
        
        Parámetros:
        -----------
        X_nuevos : list
            Lista de nuevos mensajes preprocesados
        y_nuevos : list
            Lista de etiquetas correspondientes a los nuevos mensajes
            
        Ejemplo:
        --------
        >>> modelo.entrenar(X_train, y_train)
        >>> modelo.continuar_entrenamiento(X_nuevos, y_nuevos)
        """
        if not self.entrenado:
            raise ValueError("El modelo debe ser entrenado primero antes de continuar el entrenamiento")
        
        if len(X_nuevos) != len(y_nuevos):
            raise ValueError("X_nuevos y y_nuevos deben tener la misma longitud")
        
        print(f"Continuando entrenamiento con {len(X_nuevos)} nuevos ejemplos...")
        
        # Contar nuevos mensajes por clase
        nuevos_spam = sum(1 for etq in y_nuevos if etq == 'spam')
        nuevos_ham = len(y_nuevos) - nuevos_spam
        
        # Actualizar contadores de palabras
        for mensaje, etiqueta in zip(X_nuevos, y_nuevos):
            if etiqueta == 'spam':
                self.spam_words.update(mensaje)
                self.total_spam_words += len(mensaje)
            else:  # ham
                self.ham_words.update(mensaje)
                self.total_ham_words += len(mensaje)
            
            # Agregar nuevas palabras al vocabulario
            self.vocabulario.update(mensaje)
        
        # Recalcular probabilidades a priori
        # Necesitamos el total original + nuevos
        # Para esto, estimamos el total original basado en las probabilidades actuales
        # Si P(spam) = spam_count / total, entonces total_original = spam_count / P(spam)
        if self.prob_spam > 0:
            total_original_estimado = int(self.total_spam_words / self.prob_spam) if self.prob_spam > 0 else 0
        else:
            total_original_estimado = self.total_spam_words + self.total_ham_words
        
        total_nuevo = total_original_estimado + len(y_nuevos)
        spam_total_estimado = int(total_original_estimado * self.prob_spam) + nuevos_spam
        ham_total_estimado = total_original_estimado - spam_total_estimado + nuevos_ham
        
        self.prob_spam = spam_total_estimado / total_nuevo if total_nuevo > 0 else 0.5
        self.prob_ham = ham_total_estimado / total_nuevo if total_nuevo > 0 else 0.5
        
        # Recalcular probabilidades condicionales con el nuevo vocabulario
        tamaño_vocabulario = len(self.vocabulario)
        
        # Actualizar log-probabilidades para todas las palabras
        for palabra in self.vocabulario:
            frecuencia_spam = self.spam_words.get(palabra, 0)
            prob = (frecuencia_spam + self.alpha) / (self.total_spam_words + self.alpha * tamaño_vocabulario)
            self.log_prob_spam_words[palabra] = math.log(prob)
            
            frecuencia_ham = self.ham_words.get(palabra, 0)
            prob = (frecuencia_ham + self.alpha) / (self.total_ham_words + self.alpha * tamaño_vocabulario)
            self.log_prob_ham_words[palabra] = math.log(prob)
        
        # Actualizar log-probabilidades a priori
        self.log_prob_spam = math.log(self.prob_spam) if self.prob_spam > 0 else float('-inf')
        self.log_prob_ham = math.log(self.prob_ham) if self.prob_ham > 0 else float('-inf')
        
        print(f"✓ Entrenamiento continuado exitosamente")
        print(f"  - Nuevos ejemplos spam: {nuevos_spam}")
        print(f"  - Nuevos ejemplos ham: {nuevos_ham}")
        print(f"  - Vocabulario actualizado: {len(self.vocabulario)} palabras")
        print(f"  - Nuevas probabilidades: P(spam)={self.prob_spam:.4f}, P(ham)={self.prob_ham:.4f}")


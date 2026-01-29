"""
Módulo de Evaluación del Modelo
=================================

Este módulo contiene funciones para evaluar el desempeño del clasificador
de spam usando diferentes métricas de evaluación estándar en aprendizaje automático.

Las métricas implementadas son:
- Accuracy (Exactitud)
- Precision (Precisión)
- Recall (Sensibilidad)
- F1-score
- Matriz de Confusión
"""

from collections import Counter


def matriz_confusion(y_true, y_pred):
    """
    Calcula la matriz de confusión para clasificación binaria.
    
    La matriz de confusión muestra cómo se clasificaron realmente los
    mensajes comparados con las etiquetas verdaderas:
    
    |                    | Predicho Spam | Predicho Ham |
    |--------------------|---------------|--------------|
    | Realmente Spam     | TP (True Positive)  | FN (False Negative) |
    | Realmente Ham      | FP (False Positive) | TN (True Negative)  |
    
    TP (True Positive):  Spam correctamente identificado como spam
    TN (True Negative):  Ham correctamente identificado como ham
    FP (False Positive): Ham incorrectamente identificado como spam
    FN (False Negative): Spam incorrectamente identificado como ham
    
    Parámetros:
    -----------
    y_true : list
        Lista de etiquetas verdaderas ('spam' o 'ham')
    y_pred : list
        Lista de etiquetas predichas ('spam' o 'ham')
        
    Retorna:
    --------
    dict
        Diccionario con las métricas de la matriz de confusión:
        {
            'TP': True Positives,
            'TN': True Negatives,
            'FP': False Positives,
            'FN': False Negatives
        }
    """
    # Inicializar contadores
    TP = 0  # True Positive: spam predicho correctamente
    TN = 0  # True Negative: ham predicho correctamente
    FP = 0  # False Positive: ham predicho como spam (error tipo I)
    FN = 0  # False Negative: spam predicho como ham (error tipo II)
    
    # Contar cada tipo de predicción
    for verdadero, predicho in zip(y_true, y_pred):
        if verdadero == 'spam' and predicho == 'spam':
            TP += 1
        elif verdadero == 'ham' and predicho == 'ham':
            TN += 1
        elif verdadero == 'ham' and predicho == 'spam':
            FP += 1  # Error: clasificamos ham como spam
        elif verdadero == 'spam' and predicho == 'ham':
            FN += 1  # Error: clasificamos spam como ham
    
    return {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN
    }


def calcular_accuracy(y_true, y_pred):
    """
    Calcula la exactitud (accuracy) del modelo.
    
    La exactitud es la proporción de predicciones correctas sobre el total:
    
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    Es decir, de todos los mensajes, ¿cuántos fueron clasificados correctamente?
    
    Ventajas:
    - Fácil de entender
    - Buena métrica general cuando las clases están balanceadas
    
    Desventajas:
    - Puede ser engañosa si hay desbalance de clases
    - No distingue entre tipos de errores
    
    Parámetros:
    -----------
    y_true : list
        Lista de etiquetas verdaderas ('spam' o 'ham')
    y_pred : list
        Lista de etiquetas predichas ('spam' o 'ham')
        
    Retorna:
    --------
    float
        Exactitud del modelo (entre 0.0 y 1.0, donde 1.0 es perfecto)
        
    Ejemplo:
    --------
    >>> y_true = ['spam', 'ham', 'spam', 'ham']
    >>> y_pred = ['spam', 'ham', 'ham', 'ham']
    >>> calcular_accuracy(y_true, y_pred)
    0.75  # 3 de 4 correctos
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true y y_pred deben tener la misma longitud")
    
    if len(y_true) == 0:
        return 0.0
    
    # Contar cuántas predicciones fueron correctas
    correctas = sum(1 for verdadero, predicho in zip(y_true, y_pred) if verdadero == predicho)
    
    # Accuracy = predicciones correctas / total de predicciones
    accuracy = correctas / len(y_true)
    
    return accuracy


def calcular_precision(y_true, y_pred):
    """
    Calcula la precisión (precision) del modelo.
    
    La precisión mide: de todos los mensajes que el modelo predijo como spam,
    ¿cuántos realmente eran spam?
    
    Precision = TP / (TP + FP)
    
    Es decir, cuando el modelo dice "esto es spam", ¿qué tan a menudo tiene razón?
    
    Una precisión alta significa que cuando marcamos algo como spam,
    generalmente es correcto (pocos falsos positivos).
    
    Parámetros:
    -----------
    y_true : list
        Lista de etiquetas verdaderas ('spam' o 'ham')
    y_pred : list
        Lista de etiquetas predichas ('spam' o 'ham')
        
    Retorna:
    --------
    float
        Precisión del modelo (entre 0.0 y 1.0)
        
    Ejemplo:
    --------
    >>> y_true = ['spam', 'spam', 'ham', 'ham']
    >>> y_pred = ['spam', 'spam', 'spam', 'ham']
    >>> calcular_precision(y_true, y_pred)
    0.67  # 2 TP / (2 TP + 1 FP) = 2/3
    """
    matriz = matriz_confusion(y_true, y_pred)
    TP = matriz['TP']
    FP = matriz['FP']
    
    # Si no hay predicciones positivas (spam), no podemos calcular precisión
    if TP + FP == 0:
        return 0.0
    
    # Precision = True Positives / (True Positives + False Positives)
    precision = TP / (TP + FP)
    
    return precision


def calcular_recall(y_true, y_pred):
    """
    Calcula la sensibilidad (recall) del modelo.
    
    El recall mide: de todos los mensajes que realmente son spam,
    ¿cuántos logró identificar el modelo?
    
    Recall = TP / (TP + FN)
    
    También se llama "Sensibilidad" o "Tasa de Verdaderos Positivos".
    
    Un recall alto significa que capturamos la mayoría del spam
    (pocos falsos negativos - no dejamos pasar mucho spam).
    
    Parámetros:
    -----------
    y_true : list
        Lista de etiquetas verdaderas ('spam' o 'ham')
    y_pred : list
        Lista de etiquetas predichas ('spam' o 'ham')
        
    Retorna:
    --------
    float
        Recall del modelo (entre 0.0 y 1.0)
        
    Ejemplo:
    --------
    >>> y_true = ['spam', 'spam', 'spam', 'ham']
    >>> y_pred = ['spam', 'ham', 'spam', 'ham']
    >>> calcular_recall(y_true, y_pred)
    0.67  # 2 TP / (2 TP + 1 FN) = 2/3
    """
    matriz = matriz_confusion(y_true, y_pred)
    TP = matriz['TP']
    FN = matriz['FN']
    
    # Si no hay casos positivos reales (spam), no podemos calcular recall
    if TP + FN == 0:
        return 0.0
    
    # Recall = True Positives / (True Positives + False Negatives)
    recall = TP / (TP + FN)
    
    return recall


def calcular_f1_score(y_true, y_pred):
    """
    Calcula el F1-score del modelo.
    
    El F1-score es la media armónica entre precisión y recall:
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    Es una métrica balanceada que combina precisión y recall en un solo número.
    
    ¿Por qué media armónica y no aritmética?
    - La media armónica penaliza más cuando una de las dos métricas es muy baja
    - Si precision o recall es muy bajo, el F1-score también será bajo
    - Esto nos ayuda a encontrar un balance entre ambos
    
    El F1-score es útil cuando queremos un balance entre:
    - No marcar demasiados correos legítimos como spam (alta precisión)
    - No dejar pasar demasiado spam (alto recall)
    
    Parámetros:
    -----------
    y_true : list
        Lista de etiquetas verdaderas ('spam' o 'ham')
    y_pred : list
        Lista de etiquetas predichas ('spam' o 'ham')
        
    Retorna:
    --------
    float
        F1-score del modelo (entre 0.0 y 1.0)
        
    Ejemplo:
    --------
    >>> precision = 0.8
    >>> recall = 0.75
    >>> f1 = 2 * (0.8 * 0.75) / (0.8 + 0.75)  # = 0.774
    """
    precision = calcular_precision(y_true, y_pred)
    recall = calcular_recall(y_true, y_pred)
    
    # Si ambas métricas son 0, el F1-score es 0
    if precision + recall == 0:
        return 0.0
    
    # F1-score = 2 * (Precision * Recall) / (Precision + Recall)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return f1_score


def evaluar_modelo(modelo, X_test, y_test):
    """
    Evalúa un modelo completo calculando todas las métricas disponibles.
    
    Esta función realiza predicciones sobre el conjunto de prueba y calcula
    todas las métricas de evaluación: accuracy, precision, recall, F1-score
    y la matriz de confusión.
    
    Parámetros:
    -----------
    modelo : NaiveBayesSpamDetector
        El modelo entrenado que se desea evaluar
    X_test : list
        Lista de mensajes de prueba (cada uno es una lista de tokens)
    y_test : list
        Lista de etiquetas verdaderas correspondientes
        
    Retorna:
    --------
    dict
        Diccionario con todas las métricas:
        {
            'accuracy': float,
            'precision': float,
            'recall': float,
            'f1_score': float,
            'matriz_confusion': {
                'TP': int,
                'TN': int,
                'FP': int,
                'FN': int
            }
        }
    """
    # Realizar predicciones sobre todos los mensajes de prueba
    y_pred = []
    for mensaje in X_test:
        prediccion = modelo.predecir(mensaje)
        y_pred.append(prediccion)
    
    # Calcular todas las métricas
    accuracy = calcular_accuracy(y_test, y_pred)
    precision = calcular_precision(y_test, y_pred)
    recall = calcular_recall(y_test, y_pred)
    f1_score = calcular_f1_score(y_test, y_pred)
    matriz = matriz_confusion(y_test, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'matriz_confusion': matriz
    }


def imprimir_resultados(resultados):
    """
    Imprime los resultados de la evaluación de forma legible.
    
    Parámetros:
    -----------
    resultados : dict
        Diccionario con los resultados de evaluar_modelo()
    """
    print("\n" + "="*60)
    print("RESULTADOS DE LA EVALUACIÓN")
    print("="*60)
    
    print(f"\n📊 Métricas de Desempeño:\n")
    print(f"  Accuracy  (Exactitud):    {resultados['accuracy']:.4f} ({resultados['accuracy']*100:.2f}%)")
    print(f"  Precision (Precisión):    {resultados['precision']:.4f} ({resultados['precision']*100:.2f}%)")
    print(f"  Recall    (Sensibilidad): {resultados['recall']:.4f} ({resultados['recall']*100:.2f}%)")
    print(f"  F1-Score:                 {resultados['f1_score']:.4f} ({resultados['f1_score']*100:.2f}%)")
    
    matriz = resultados['matriz_confusion']
    print(f"\n📋 Matriz de Confusión:\n")
    print(f"                    Predicho")
    print(f"                  Spam    Ham")
    print(f"  Realmente Spam   {matriz['TP']:4d}   {matriz['FN']:4d}")
    print(f"  Realmente Ham    {matriz['FP']:4d}   {matriz['TN']:4d}")
    
    print("\n" + "="*60 + "\n")


"""
Módulo de Análisis Estadístico
================================

Este módulo contiene funciones para realizar análisis estadísticos
detallados del modelo y los datos.
"""

import json
from collections import Counter
from datetime import datetime


def analizar_distribucion_datos(mensajes, etiquetas):
    """
    Realiza un análisis estadístico de la distribución de los datos.
    
    Parámetros:
    -----------
    mensajes : list
        Lista de mensajes
    etiquetas : list
        Lista de etiquetas
        
    Retorna:
    --------
    dict
        Diccionario con estadísticas detalladas
    """
    spam_count = sum(1 for e in etiquetas if e == 'spam')
    ham_count = len(etiquetas) - spam_count
    total = len(etiquetas)
    
    # Calcular longitudes de mensajes
    longitudes_spam = [len(msg.split()) for msg, etq in zip(mensajes, etiquetas) if etq == 'spam']
    longitudes_ham = [len(msg.split()) for msg, etq in zip(mensajes, etiquetas) if etq == 'ham']
    
    estadisticas = {
        'total_mensajes': total,
        'distribucion_clases': {
            'spam': {
                'cantidad': spam_count,
                'porcentaje': (spam_count / total) * 100 if total > 0 else 0
            },
            'ham': {
                'cantidad': ham_count,
                'porcentaje': (ham_count / total) * 100 if total > 0 else 0
            }
        },
        'longitud_promedio_mensajes': {
            'spam': sum(longitudes_spam) / len(longitudes_spam) if longitudes_spam else 0,
            'ham': sum(longitudes_ham) / len(longitudes_ham) if longitudes_ham else 0,
            'general': sum(len(msg.split()) for msg in mensajes) / len(mensajes) if mensajes else 0
        },
        'balance': 'balanceado' if abs(spam_count - ham_count) / total < 0.1 else 'desbalanceado'
    }
    
    return estadisticas


def analizar_errores(modelo, X_test, y_test, mensajes_test):
    """
    Analiza los errores de clasificación del modelo.
    
    Parámetros:
    -----------
    modelo : NaiveBayesSpamDetector
        Modelo entrenado
    X_test : list
        Mensajes de prueba preprocesados
    y_test : list
        Etiquetas verdaderas
    mensajes_test : list
        Mensajes originales de prueba
        
    Retorna:
    --------
    dict
        Análisis de errores con ejemplos
    """
    errores = {
        'falsos_positivos': [],  # Ham clasificado como spam
        'falsos_negativos': []   # Spam clasificado como ham
    }
    
    for mensaje_preproc, mensaje_original, etiqueta_real in zip(X_test, mensajes_test, y_test):
        prediccion = modelo.predecir(mensaje_preproc)
        probabilidades = modelo.predecir_proba(mensaje_preproc)
        
        if prediccion != etiqueta_real:
            error_info = {
                'mensaje': mensaje_original,
                'etiqueta_real': etiqueta_real,
                'prediccion': prediccion,
                'probabilidad_spam': probabilidades['spam'],
                'probabilidad_ham': probabilidades['ham']
            }
            
            if etiqueta_real == 'ham' and prediccion == 'spam':
                errores['falsos_positivos'].append(error_info)
            elif etiqueta_real == 'spam' and prediccion == 'ham':
                errores['falsos_negativos'].append(error_info)
    
    return errores


def generar_reporte_completo(resultados, estadisticas_datos, errores, 
                             palabras_importantes, ruta_archivo='resultados/reporte_completo.json'):
    """
    Genera un reporte completo en formato JSON con todos los análisis.
    
    Parámetros:
    -----------
    resultados : dict
        Resultados de la evaluación
    estadisticas_datos : dict
        Estadísticas de los datos
    errores : dict
        Análisis de errores
    palabras_importantes : dict
        Palabras importantes por clase
    ruta_archivo : str
        Ruta donde guardar el reporte
    """
    import os
    
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(ruta_archivo) if os.path.dirname(ruta_archivo) else '.', exist_ok=True)
    
    reporte = {
        'fecha_generacion': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'modelo': {
            'tipo': 'Naive Bayes',
            'algoritmo': 'Clasificador Bayesiano Ingenuo',
            'suavizado': 'Laplace (alpha=1.0)'
        },
        'datos': estadisticas_datos,
        'resultados_evaluacion': {
            'metricas': {
                'accuracy': resultados['accuracy'],
                'precision': resultados['precision'],
                'recall': resultados['recall'],
                'f1_score': resultados['f1_score']
            },
            'matriz_confusion': resultados['matriz_confusion']
        },
        'analisis_errores': {
            'total_falsos_positivos': len(errores['falsos_positivos']),
            'total_falsos_negativos': len(errores['falsos_negativos']),
            'tasa_error_fp': len(errores['falsos_positivos']) / len(errores['falsos_positivos'] + errores['falsos_negativos']) if (errores['falsos_positivos'] + errores['falsos_negativos']) else 0,
            'tasa_error_fn': len(errores['falsos_negativos']) / len(errores['falsos_positivos'] + errores['falsos_negativos']) if (errores['falsos_positivos'] + errores['falsos_negativos']) else 0,
            'ejemplos_falsos_positivos': errores['falsos_positivos'][:5],  # Primeros 5
            'ejemplos_falsos_negativos': errores['falsos_negativos'][:5]
        },
        'palabras_importantes': {
            'spam': [{'palabra': p[0], 'diferencia': p[1]} for p in palabras_importantes['spam'][:20]],
            'ham': [{'palabra': p[0], 'diferencia': p[1]} for p in palabras_importantes['ham'][:20]]
        },
        'interpretacion': {
            'desempeno_general': interpretar_desempeno(resultados),
            'recomendaciones': generar_recomendaciones(resultados, estadisticas_datos, errores)
        }
    }
    
    with open(ruta_archivo, 'w', encoding='utf-8') as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Reporte completo guardado en: {ruta_archivo}")
    return reporte


def interpretar_desempeno(resultados):
    """
    Interpreta el desempeño del modelo basándose en las métricas.
    
    Parámetros:
    -----------
    resultados : dict
        Resultados de la evaluación
        
    Retorna:
    --------
    dict
        Interpretación del desempeño
    """
    accuracy = resultados['accuracy']
    precision = resultados['precision']
    recall = resultados['recall']
    f1 = resultados['f1_score']
    
    interpretacion = {
        'nivel_general': '',
        'fortalezas': [],
        'debilidades': [],
        'aplicabilidad': ''
    }
    
    # Determinar nivel general
    promedio = (accuracy + precision + recall + f1) / 4
    if promedio >= 0.9:
        interpretacion['nivel_general'] = 'Excelente'
        interpretacion['aplicabilidad'] = 'El modelo está listo para uso en producción'
    elif promedio >= 0.8:
        interpretacion['nivel_general'] = 'Bueno'
        interpretacion['aplicabilidad'] = 'El modelo tiene buen desempeño, con margen de mejora'
    elif promedio >= 0.7:
        interpretacion['nivel_general'] = 'Aceptable'
        interpretacion['aplicabilidad'] = 'El modelo funciona pero necesita mejoras'
    else:
        interpretacion['nivel_general'] = 'Necesita Mejoras'
        interpretacion['aplicabilidad'] = 'El modelo requiere ajustes significativos'
    
    # Identificar fortalezas
    if accuracy >= 0.85:
        interpretacion['fortalezas'].append('Alta exactitud general')
    if precision >= 0.85:
        interpretacion['fortalezas'].append('Baja tasa de falsos positivos (no molesta a usuarios)')
    if recall >= 0.85:
        interpretacion['fortalezas'].append('Alta capacidad de detectar spam')
    if f1 >= 0.85:
        interpretacion['fortalezas'].append('Balance adecuado entre precisión y recall')
    
    # Identificar debilidades
    if precision < 0.7:
        interpretacion['debilidades'].append('Muchos correos legítimos marcados como spam')
    if recall < 0.7:
        interpretacion['debilidades'].append('Mucho spam no detectado')
    if accuracy < 0.75:
        interpretacion['debilidades'].append('Baja exactitud general')
    
    return interpretacion


def generar_recomendaciones(resultados, estadisticas_datos, errores):
    """
    Genera recomendaciones para mejorar el modelo.
    
    Parámetros:
    -----------
    resultados : dict
        Resultados de la evaluación
    estadisticas_datos : dict
        Estadísticas de los datos
    errores : dict
        Análisis de errores
        
    Retorna:
    --------
    list
        Lista de recomendaciones
    """
    recomendaciones = []
    
    # Recomendaciones basadas en balance de datos
    if estadisticas_datos['balance'] == 'desbalanceado':
        recomendaciones.append('Balancear el dataset: agregar más ejemplos de la clase minoritaria')
    
    # Recomendaciones basadas en métricas
    if resultados['precision'] < 0.8:
        recomendaciones.append('Mejorar precisión: revisar preprocesamiento y agregar más ejemplos de ham')
    
    if resultados['recall'] < 0.8:
        recomendaciones.append('Mejorar recall: agregar más ejemplos de spam al entrenamiento')
    
    # Recomendaciones basadas en errores
    if len(errores['falsos_positivos']) > len(errores['falsos_negativos']):
        recomendaciones.append('Reducir falsos positivos: ajustar umbral de decisión o mejorar características de ham')
    
    if len(errores['falsos_negativos']) > len(errores['falsos_positivos']):
        recomendaciones.append('Reducir falsos negativos: mejorar detección de patrones de spam')
    
    # Recomendaciones generales
    if estadisticas_datos['total_mensajes'] < 200:
        recomendaciones.append('Aumentar el tamaño del dataset para mejorar la generalización')
    
    recomendaciones.append('Considerar técnicas avanzadas: n-gramas, TF-IDF, o modelos más complejos')
    recomendaciones.append('Validar manualmente las etiquetas del dataset para asegurar calidad')
    
    return recomendaciones


def imprimir_analisis_completo(estadisticas_datos, resultados, errores):
    """
    Imprime un análisis completo y profesional de los resultados.
    
    Parámetros:
    -----------
    estadisticas_datos : dict
        Estadísticas de los datos
    resultados : dict
        Resultados de la evaluación
    errores : dict
        Análisis de errores
    """
    print("\n" + "="*70)
    print("ANÁLISIS ESTADÍSTICO COMPLETO")
    print("="*70)
    
    # Análisis de datos
    print("\n📊 ANÁLISIS DEL DATASET:")
    print("-"*70)
    print(f"Total de mensajes: {estadisticas_datos['total_mensajes']}")
    print(f"Distribución:")
    print(f"  - Spam: {estadisticas_datos['distribucion_clases']['spam']['cantidad']} "
          f"({estadisticas_datos['distribucion_clases']['spam']['porcentaje']:.1f}%)")
    print(f"  - Ham: {estadisticas_datos['distribucion_clases']['ham']['cantidad']} "
          f"({estadisticas_datos['distribucion_clases']['ham']['porcentaje']:.1f}%)")
    print(f"Balance: {estadisticas_datos['balance'].upper()}")
    print(f"Longitud promedio de mensajes:")
    print(f"  - Spam: {estadisticas_datos['longitud_promedio_mensajes']['spam']:.1f} palabras")
    print(f"  - Ham: {estadisticas_datos['longitud_promedio_mensajes']['ham']:.1f} palabras")
    
    # Análisis de errores
    print("\n🔍 ANÁLISIS DE ERRORES:")
    print("-"*70)
    total_errores = len(errores['falsos_positivos']) + len(errores['falsos_negativos'])
    print(f"Total de errores: {total_errores}")
    print(f"  - Falsos Positivos (Ham → Spam): {len(errores['falsos_positivos'])}")
    print(f"  - Falsos Negativos (Spam → Ham): {len(errores['falsos_negativos'])}")
    
    if total_errores > 0:
        tasa_fp = (len(errores['falsos_positivos']) / total_errores) * 100
        tasa_fn = (len(errores['falsos_negativos']) / total_errores) * 100
        print(f"\nDistribución de errores:")
        print(f"  - Falsos Positivos: {tasa_fp:.1f}%")
        print(f"  - Falsos Negativos: {tasa_fn:.1f}%")
    
    # Interpretación
    interpretacion = interpretar_desempeno(resultados)
    print("\n📈 INTERPRETACIÓN DEL DESEMPEÑO:")
    print("-"*70)
    print(f"Nivel General: {interpretacion['nivel_general']}")
    print(f"Aplicabilidad: {interpretacion['aplicabilidad']}")
    
    if interpretacion['fortalezas']:
        print(f"\nFortalezas:")
        for fortaleza in interpretacion['fortalezas']:
            print(f"  ✓ {fortaleza}")
    
    if interpretacion['debilidades']:
        print(f"\nÁreas de Mejora:")
        for debilidad in interpretacion['debilidades']:
            print(f"  ⚠ {debilidad}")
    
    # Recomendaciones
    recomendaciones = generar_recomendaciones(resultados, estadisticas_datos, errores)
    print("\n💡 RECOMENDACIONES:")
    print("-"*70)
    for i, rec in enumerate(recomendaciones, 1):
        print(f"  {i}. {rec}")
    
    print("\n" + "="*70 + "\n")

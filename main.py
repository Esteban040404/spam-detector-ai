import csv
import random
import os
from preprocesamiento import preprocesar_dataset
from modelo import NaiveBayesSpamDetector
from evaluacion import evaluar_modelo, imprimir_resultados
try:
    from visualizaciones import generar_reporte_visual
    VISUALIZACIONES_DISPONIBLES = True
except ImportError as e:
    VISUALIZACIONES_DISPONIBLES = False
    print(f"⚠ Advertencia: Visualizaciones no disponibles: {e}")
    print("   El proyecto funcionará pero sin gráficos. Instala con: pip install matplotlib seaborn numpy")
from analisis import (
    analizar_distribucion_datos,
    analizar_errores,
    generar_reporte_completo,
    imprimir_analisis_completo
)


def cargar_datos(archivo_csv):
    """
    Carga los datos desde un archivo CSV.
    
    El archivo CSV debe tener las columnas: id, mensaje, etiqueta
    donde etiqueta es 'spam' o 'ham'.
    
    Parámetros:
    -----------
    archivo_csv : str
        Ruta al archivo CSV con los datos
        
    Retorna:
    --------
    tuple: (mensajes, etiquetas)
        - mensajes: lista de strings con los mensajes
        - etiquetas: lista de strings con las etiquetas ('spam' o 'ham')
    """
    mensajes = []
    etiquetas = []
    
    try:
        with open(archivo_csv, 'r', encoding='utf-8') as archivo:
            lector = csv.DictReader(archivo)
            
            for fila in lector:
                mensaje = fila['mensaje'].strip()
                etiqueta = fila['etiqueta'].strip().lower()
                
                # Validar que la etiqueta sea válida
                if etiqueta not in ['spam', 'ham']:
                    print(f"Advertencia: Etiqueta inválida '{etiqueta}' en mensaje '{mensaje[:50]}...'")
                    continue
                
                mensajes.append(mensaje)
                etiquetas.append(etiqueta)
        
        print(f"✓ Datos cargados: {len(mensajes)} mensajes")
        return mensajes, etiquetas
    
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {archivo_csv}")
        raise
    except Exception as e:
        print(f"Error al cargar datos: {e}")
        raise


def dividir_datos(mensajes, etiquetas, porcentaje_entrenamiento=0.8, semilla=None):
    """
    Divide los datos en conjuntos de entrenamiento y prueba.
    
    Parámetros:
    -----------
    mensajes : list
        Lista de mensajes
    etiquetas : list
        Lista de etiquetas correspondientes
    porcentaje_entrenamiento : float, default=0.8
        Proporción de datos para entrenamiento (0.0 a 1.0)
        El resto se usa para prueba
    semilla : int, optional
        Semilla para el generador aleatorio (para reproducibilidad)
        
    Retorna:
    --------
    tuple: (X_train, X_test, y_train, y_test)
        - X_train: mensajes de entrenamiento (listas de tokens)
        - X_test: mensajes de prueba (listas de tokens)
        - y_train: etiquetas de entrenamiento
        - y_test: etiquetas de prueba
    """
    if semilla is not None:
        random.seed(semilla)
    
    # Crear lista de índices y mezclarla
    indices = list(range(len(mensajes)))
    random.shuffle(indices)
    
    # Calcular punto de división
    punto_division = int(len(mensajes) * porcentaje_entrenamiento)
    
    # Dividir índices
    indices_train = indices[:punto_division]
    indices_test = indices[punto_division:]
    
    # Crear conjuntos de entrenamiento
    mensajes_train = [mensajes[i] for i in indices_train]
    etiquetas_train = [etiquetas[i] for i in indices_train]
    
    # Crear conjuntos de prueba
    mensajes_test = [mensajes[i] for i in indices_test]
    etiquetas_test = [etiquetas[i] for i in indices_test]
    
    print(f"✓ Datos divididos:")
    print(f"  - Entrenamiento: {len(mensajes_train)} mensajes")
    print(f"  - Prueba: {len(mensajes_test)} mensajes")
    
    return mensajes_train, mensajes_test, etiquetas_train, etiquetas_test


def main():
    """
    Función principal que ejecuta todo el pipeline del detector de spam.
    """
    print("="*60)
    print("DETECTOR DE SPAM - NAIVE BAYES")
    print("="*60)
    print("\nIniciando pipeline de entrenamiento y evaluación...\n")
    
    # Paso 1: Cargar datos
    # Intentar cargar dataset grande primero, si no existe usar el pequeño
    if os.path.exists('datos_grande.csv'):
        print("📂 Paso 1: Cargando datos desde datos_grande.csv...")
        mensajes, etiquetas = cargar_datos('datos_grande.csv')
    elif os.path.exists('datos.csv'):
        print("📂 Paso 1: Cargando datos desde datos.csv...")
        mensajes, etiquetas = cargar_datos('datos.csv')
    else:
        print("❌ Error: No se encontró ningún archivo de datos.")
        print("   Ejecuta primero: python generar_dataset_grande.py")
        return
    
    # Mostrar estadísticas de los datos
    spam_count = sum(1 for e in etiquetas if e == 'spam')
    ham_count = len(etiquetas) - spam_count
    print(f"  - Mensajes spam: {spam_count}")
    print(f"  - Mensajes ham: {ham_count}\n")
    
    # Paso 2: Dividir datos en entrenamiento y prueba
    print("✂️  Paso 2: Dividiendo datos en entrenamiento (80%) y prueba (20%)...")
    mensajes_train, mensajes_test, etiquetas_train, etiquetas_test = dividir_datos(
        mensajes, etiquetas, porcentaje_entrenamiento=0.8, semilla=42
    )
    print()
    
    # Paso 3: Preprocesar datos
    print("🔧 Paso 3: Preprocesando mensajes...")
    print("  - Normalizando texto (minúsculas, sin puntuación)")
    print("  - Tokenizando (dividiendo en palabras)")
    print("  - Eliminando stopwords (palabras comunes)")
    
    # Preprocesar mensajes de entrenamiento
    # preprocesar_dataset retorna (mensajes_preprocesados, vocabulario, vectores)
    # En este caso, solo necesitamos los mensajes preprocesados
    mensajes_train_preproc, vocab_train, _ = preprocesar_dataset(mensajes_train)
    
    # Preprocesar mensajes de prueba usando el mismo vocabulario de entrenamiento
    # (importante: usar el vocabulario del entrenamiento, no crear uno nuevo)
    from preprocesamiento import preprocesar_mensaje
    mensajes_test_preproc = [preprocesar_mensaje(msg) for msg in mensajes_test]
    
    print(f"  - Vocabulario creado: {len(vocab_train)} palabras únicas")
    print()
    
    # Paso 4: Entrenar modelo
    print("🎓 Paso 4: Entrenando modelo Naive Bayes...")
    print("  - Calculando probabilidades a priori P(spam) y P(ham)")
    print("  - Contando frecuencias de palabras por clase")
    print("  - Aplicando suavizado de Laplace (alpha=1.0)")
    
    # Entrenar modelo (siempre entrenar con los datos actuales)
    # El modelo se guardará al final para preservar el conocimiento
    modelo = NaiveBayesSpamDetector(alpha=1.0)
    modelo.entrenar(mensajes_train_preproc, etiquetas_train)
    
    print(f"  - Modelo entrenado exitosamente")
    print(f"  - Probabilidad a priori P(spam) = {modelo.prob_spam:.4f}")
    print(f"  - Probabilidad a priori P(ham) = {modelo.prob_ham:.4f}")
    
    # Guardar el modelo entrenado para preservar el conocimiento
    ruta_modelo = 'modelos/modelo_entrenado.pkl'
    os.makedirs('modelos', exist_ok=True)
    modelo.guardar(ruta_modelo)
    print()
    
    # Paso 5: Evaluar modelo
    print("📊 Paso 5: Evaluando modelo con datos de prueba...")
    resultados = evaluar_modelo(modelo, mensajes_test_preproc, etiquetas_test)
    
    # Imprimir resultados de forma legible
    imprimir_resultados(resultados)
    
    # Paso 6: Ejemplos de predicción
    print("🔍 Paso 6: Ejemplos de predicción en mensajes de prueba:\n")
    print("-"*60)
    
    # Seleccionar algunos ejemplos aleatorios para mostrar
    ejemplos_indices = random.sample(range(len(mensajes_test)), min(5, len(mensajes_test)))
    
    for i, idx in enumerate(ejemplos_indices, 1):
        mensaje_original = mensajes_test[idx]
        etiqueta_real = etiquetas_test[idx]
        mensaje_preproc = mensajes_test_preproc[idx]
        
        # Hacer predicción
        prediccion = modelo.predecir(mensaje_preproc)
        probabilidades = modelo.predecir_proba(mensaje_preproc)
        
        # Determinar si la predicción fue correcta
        correcto = "✓" if prediccion == etiqueta_real else "✗"
        
        print(f"\nEjemplo {i}:")
        print(f"  Mensaje: {mensaje_original[:70]}...")
        print(f"  Real: {etiqueta_real.upper()}")
        print(f"  Predicción: {prediccion.upper()} {correcto}")
        print(f"  Probabilidades: Spam={probabilidades['spam']:.3f}, Ham={probabilidades['ham']:.3f}")
    
    print("\n" + "-"*60 + "\n")
    
    # Paso 7: Análisis de palabras importantes
    print("🔑 Paso 7: Palabras más importantes para cada clase:\n")
    palabras_importantes = modelo.obtener_palabras_importantes(top_n=15)
    
    print("Palabras más características de SPAM:")
    for i, (palabra, diferencia) in enumerate(palabras_importantes['spam'][:10], 1):
        print(f"  {i:2d}. {palabra:20s} (diferencia: {diferencia:.6f})")
    
    print("\nPalabras más características de HAM:")
    for i, (palabra, diferencia) in enumerate(palabras_importantes['ham'][:10], 1):
        print(f"  {i:2d}. {palabra:20s} (diferencia: {diferencia:.6f})")
    
    # Paso 8: Análisis estadístico completo
    print("\n" + "="*60)
    print("📊 Paso 8: Análisis estadístico completo...")
    print("="*60)
    
    estadisticas_datos = analizar_distribucion_datos(mensajes, etiquetas)
    errores = analizar_errores(modelo, mensajes_test_preproc, etiquetas_test, mensajes_test)
    
    imprimir_analisis_completo(estadisticas_datos, resultados, errores)
    
    # Paso 9: Generar visualizaciones (si están disponibles)
    if VISUALIZACIONES_DISPONIBLES:
        print("🎨 Paso 9: Generando visualizaciones profesionales...")
        print("-"*60)
        
        # Crear directorio de resultados
        directorio_resultados = 'resultados'
        os.makedirs(directorio_resultados, exist_ok=True)
        
        try:
            generar_reporte_visual(
                resultados, 
                etiquetas_train, 
                etiquetas_test, 
                palabras_importantes,
                directorio=directorio_resultados
            )
        except Exception as e:
            print(f"⚠ Error al generar visualizaciones: {e}")
            print("   Continuando sin visualizaciones...")
    else:
        print("🎨 Paso 9: Visualizaciones omitidas (dependencias no instaladas)")
        print("-"*60)
        print("   Para habilitar visualizaciones, instala: pip install matplotlib seaborn numpy")
    
    # Paso 10: Generar reporte completo en JSON
    print("\n📄 Paso 10: Generando reporte completo en JSON...")
    print("-"*60)
    
    reporte_completo = generar_reporte_completo(
        resultados,
        estadisticas_datos,
        errores,
        palabras_importantes,
        ruta_archivo=os.path.join(directorio_resultados, 'reporte_completo.json')
    )
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    print("="*60)
    print(f"\n📁 Todos los resultados han sido guardados en: '{directorio_resultados}/'")
    print("\nArchivos generados:")
    print("  - metricas_desempeno.png")
    print("  - matriz_confusion.png")
    print("  - distribucion_clases.png")
    print("  - palabras_importantes.png")
    print("  - comparacion_metricas_radar.png")
    print("  - reporte_completo.json")
    
    # Ejemplos de uso del modelo
    print("\n" + "="*60)
    print("💡 EJEMPLOS DE USO DEL MODELO")
    print("="*60)
    print("\nEjemplo 1: Mensaje de spam")
    ejemplo1 = "Gana dinero rápido sin esfuerzo haciendo clic aquí"
    pred1 = modelo.predecir(ejemplo1)
    prob1 = modelo.predecir_proba(ejemplo1)
    print(f"  Mensaje: '{ejemplo1}'")
    print(f"  Predicción: {pred1.upper()}")
    print(f"  Probabilidades: Spam={prob1['spam']:.3f}, Ham={prob1['ham']:.3f}")
    
    print("\nEjemplo 2: Mensaje legítimo")
    ejemplo2 = "Reunión confirmada para mañana a las 3pm en la sala de juntas"
    pred2 = modelo.predecir(ejemplo2)
    prob2 = modelo.predecir_proba(ejemplo2)
    print(f"  Mensaje: '{ejemplo2}'")
    print(f"  Predicción: {pred2.upper()}")
    print(f"  Probabilidades: Spam={prob2['spam']:.3f}, Ham={prob2['ham']:.3f}")
    
    # Información sobre persistencia del modelo
    print("\n" + "="*60)
    print("💾 PERSISTENCIA DEL MODELO")
    print("="*60)
    print(f"\nEl modelo entrenado ha sido guardado en: '{ruta_modelo}'")
    print("Puedes cargarlo en el futuro sin necesidad de reentrenar:")
    print("  from modelo import NaiveBayesSpamDetector")
    print(f"  modelo = NaiveBayesSpamDetector.cargar('{ruta_modelo}')")
    print("\nTambién puedes continuar el entrenamiento con nuevos datos:")
    print("  modelo.continuar_entrenamiento(X_nuevos, y_nuevos)")
    print("  modelo.guardar('modelos/modelo_entrenado.pkl')")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()


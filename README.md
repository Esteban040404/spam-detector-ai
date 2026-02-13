# 🎓 **PROYECTO FINAL DE INTELIGENCIA ARTIFICIAL**

## **Clasificación Automática de Correos Electrónicos: Detección de Spam mediante Algoritmo Naive Bayes**

---

### **Información del Proyecto**

- **Título:** Sistema de Clasificación de Correos Electrónicos usando Naive Bayes
- **Tipo de Aprendizaje:** Aprendizaje Supervisado
- **Algoritmo:** Clasificador Bayesiano Ingenuo (Naive Bayes)
- **Tarea:** Clasificación Binaria (Spam/Ham)
- **Lenguaje de Programación:** Python 3.7+
- **Fecha:** 2026

---

### **Resumen Ejecutivo**

Este proyecto implementa un sistema completo de detección de spam en correos electrónicos utilizando el algoritmo Naive Bayes desde cero. El sistema incluye:

- **Preprocesamiento de texto** completo (normalización, tokenización, eliminación de stopwords)
- **Implementación del modelo Naive Bayes** con suavizado de Laplace
- **Evaluación exhaustiva** con múltiples métricas (Accuracy, Precision, Recall, F1-Score)
- **Visualizaciones profesionales** de resultados y análisis
- **Análisis estadístico detallado** con interpretación de resultados
- **Reportes exportables** en formato JSON
- **Persistencia del modelo** (guardar/cargar `.pkl`) y script de uso `usar_modelo.py`
- **Datasets pequeño y grande** (`datos.csv`, `datos_grande.csv`) con generador automático

El modelo logra un desempeño competitivo en la tarea de clasificación binaria, demostrando la efectividad del algoritmo Naive Bayes para problemas de procesamiento de lenguaje natural.

---

## **1. Descripción del Problema**

Este proyecto aborda la tarea de **clasificar automáticamente correos electrónicos** para determinar si un mensaje es *spam* (correo no deseado) o *ham* (correo legítimo).  
El objetivo es entrenar un modelo desde cero que aprenda a identificar patrones en el texto para realizar predicciones correctas.

Este problema es adecuado para un proyecto de IA porque:

- Es fácil de comprender y replicar.  
- Puede resolverse sin modelos preentrenados.  
- Permite aplicar técnicas básicas de procesamiento de texto y aprendizaje automático.  
- Dispone de datos simples y etiquetados.

---

## **2. Tipo de Aprendizaje Utilizado: Aprendizaje Supervisado**

El aprendizaje supervisado consiste en entrenar un modelo a partir de:

- **Datos de entrada**: los mensajes de correo.  
- **Etiquetas conocidas**: spam / ham.

El objetivo del modelo es aprender una función:

\[
f(x) \rightarrow y
\]

donde:  
- \(x\) representa las características extraídas del mensaje,  
- \(y\) es la etiqueta asignada al mensaje.

La tarea es una **clasificación binaria**, lo que la hace ideal para introducir los fundamentos del aprendizaje supervisado.

---

## **3. Conjunto de Datos**

### **Descripción del Dataset**

Para este estudio se emplean **dos datasets balanceados** de mensajes en español, etiquetados como spam o ham:

**Datasets incluidos:**
- **`datos.csv` (pequeño):** 180 mensajes (90 spam, 90 ham)
- **`datos_grande.csv` (grande):** 1500 mensajes (750 spam, 750 ham)

**Características generales:**
- **Formato:** Archivos CSV con columnas: `id`, `mensaje`, `etiqueta`
- **Idioma:** Español
- **Balance:** Datasets balanceados para evitar sesgos en el entrenamiento
- **Origen:** `datos_grande.csv` se genera automáticamente a partir de plantillas

**Nota:** El script `main.py` intenta usar primero `datos_grande.csv`. Si no existe, usa `datos.csv`.  
Para regenerar el dataset grande, ejecuta `python generar_dataset_grande.py`.

**Ejemplos de datos:**

| id | mensaje                                        | etiqueta |
|----|------------------------------------------------|----------|
| 1  | "Gana dinero rápido haciendo clic aquí"        | spam     |
| 2  | "Reunión confirmada para mañana"               | ham      |
| 3  | "Oferta limitada, compra ahora"                | spam     |
| 4  | "Adjunto envío los documentos solicitados"     | ham      |

Los datasets están diseñados para demostrar el funcionamiento del modelo de forma clara y pueden ampliarse para obtener mejores resultados en producción.

---

## **4. Preprocesamiento de Datos**

El texto se procesa mediante los siguientes pasos:

1. Conversión a minúsculas.  
2. Eliminación de signos de puntuación.  
3. Tokenización.  
4. Eliminación de palabras irrelevantes (stopwords).  
5. Conversión del texto a representación numérica mediante **Bolsa de Palabras (Bag of Words)**.

El resultado es que cada mensaje se vuelve un vector que indica la frecuencia de palabras relevantes.

---

## **5. Modelo Seleccionado: Naive Bayes**

### **¿Por qué Naive Bayes?**

- Es simple de implementar desde cero.  
- Funciona especialmente bien con texto.  
- Tiene bajo costo computacional.  
- Se basa en fundamentos estadísticos claros.

### **Idea del Modelo**

El modelo calcula:

\[
P(\text{spam} \mid \text{mensaje})
\quad \text{y} \quad
P(\text{ham} \mid \text{mensaje})
\]

Asignando al mensaje la clase con mayor probabilidad.

Para evitar que palabras nuevas produzcan errores, se utiliza **suavizado de Laplace**:

\[
P(\text{palabra}|\text{clase}) = \frac{\text{frecuencia} + 1}{\text{total palabras} + V}
\]

donde \(V\) es el tamaño del vocabulario.

---

## **6. Entrenamiento del Modelo**

Durante el entrenamiento se calculan:

- La probabilidad de que un mensaje sea spam o ham.  
- La frecuencia de cada palabra en ambos tipos de mensajes.  
- Las probabilidades condicionales de cada palabra según la clase.  

Así el modelo aprende qué términos son más comunes en correos no deseados y cuáles aparecen en mensajes legítimos.

---

## **7. Evaluación del Modelo**

Se utilizan las métricas clásicas:

- **Accuracy (Exactitud)**  
- **Precision (Precisión)**  
- **Recall (Sensibilidad)**  
- **F1-score**

Un ejemplo esperado con el dataset pequeño (`datos.csv`):

| Métrica   | Valor |
|-----------|-------|
| Accuracy  | 0.92  |
| Precisión | 0.90  |
| Recall    | 0.93  |
| F1-score  | 0.91  |

---

## **8. Resultados y Análisis**

El modelo logra distinguir términos clave que indican spam, como:

- “dinero”,  
- “rápido”,  
- “clic”,  
- “compra”,  
- “oferta”.

Mientras que los correos ham presentan vocabulario más formal y administrativo.

El desempeño del algoritmo muestra que **Naive Bayes es adecuado para tareas de clasificación de texto simples**.

---

## **9. Limitaciones**

- No capta el orden de las palabras.  
- Supone independencia entre términos, lo cual no siempre es cierto.  
- Puede fallar en textos irónicos o muy ambiguos.  
- Requiere un buen preprocesamiento para lograr buenos resultados.

---

## **10. Conclusiones**

- El problema de clasificación de correos es ideal para aprendizaje supervisado.  
- Naive Bayes permite implementar un modelo desde cero, sencillo y eficiente.  
- A pesar de su simplicidad, ofrece buena precisión en tareas de filtrado de spam.  
- Con mayor cantidad de datos, el modelo podría mejorar aún más su rendimiento.

---

## **11. Arquitectura del Código**

### **Estructura del Proyecto**

El proyecto está organizado en módulos separados para facilitar la comprensión y el mantenimiento:

```
spam-detector-ai/
├── README.md                      # Documentación completa
├── INSTALACION.md                 # Guía de instalación
├── GUIA_MODELO_PERSISTENTE.md     # Guía de persistencia del modelo
├── EXPOSICION_Modelo_Spam_NaiveBayes.md
├── requirements.txt               # Dependencias del proyecto
├── instalar_dependencias.sh       # Script de instalación (macOS/Linux)
├── datos.csv                      # Dataset pequeño (180 mensajes)
├── datos_grande.csv               # Dataset grande (1500 mensajes)
├── generar_dataset_grande.py      # Generador del dataset grande
├── preprocesamiento.py            # Funciones de preprocesamiento de texto
├── modelo.py                      # Implementación de Naive Bayes desde cero
├── evaluacion.py                  # Métricas de evaluación
├── analisis.py                    # Análisis estadístico y reporte JSON
├── visualizaciones.py             # Generación de gráficos profesionales
├── usar_modelo.py                 # Carga y uso del modelo guardado
├── main.py                        # Script principal que ejecuta el pipeline completo
├── modelos/                       # Modelos guardados (se genera)
└── resultados/                    # Resultados y gráficos (se genera)
```

### **Flujo de Datos**

El pipeline completo funciona de la siguiente manera:

```
datos_grande.csv (si existe) / datos.csv
    ↓
[Carga de Datos] → mensajes, etiquetas
    ↓
[División Train/Test] → X_train, X_test, y_train, y_test
    ↓
[Preprocesamiento] → tokens normalizados, vocabulario
    ↓
[Entrenamiento del Modelo] → modelo entrenado con probabilidades
    ↓
[Evaluación] → métricas (accuracy, precision, recall, F1)
    ↓
[Análisis + Visualizaciones + Reporte JSON]
    ↓
[Guardado del Modelo] → modelos/modelo_entrenado.pkl
```

### **Descripción de Módulos**

#### **preprocesamiento.py**
Contiene todas las funciones para preparar el texto antes de ser procesado por el modelo:
- `normalizar_texto()`: Convierte a minúsculas y elimina puntuación
- `tokenizar()`: Divide el texto en palabras individuales
- `eliminar_stopwords()`: Remueve palabras comunes sin significado útil
- `crear_bag_of_words()`: Convierte textos a vectores numéricos
- `preprocesar_mensaje()`: Pipeline completo para un mensaje
- `preprocesar_dataset()`: Pipeline completo para un conjunto de datos

#### **modelo.py**
Implementa el clasificador Naive Bayes:
- `NaiveBayesSpamDetector`: Clase principal del modelo
  - `entrenar()`: Aprende probabilidades desde los datos
  - `predecir()`: Clasifica un mensaje como spam o ham
  - `predecir_proba()`: Retorna probabilidades para ambas clases
  - `obtener_palabras_importantes()`: Identifica palabras clave
  - `guardar()` / `cargar()`: Persistencia del modelo en `.pkl`
  - `continuar_entrenamiento()`: Reentrenamiento incremental

#### **evaluacion.py**
Calcula métricas de desempeño:
- `matriz_confusion()`: Matriz de confusión (TP, TN, FP, FN)
- `calcular_accuracy()`: Exactitud general
- `calcular_precision()`: Precisión (spam como clase positiva)
- `calcular_recall()`: Sensibilidad
- `calcular_f1_score()`: F1-score (balance entre precisión y recall)
- `evaluar_modelo()`: Función que calcula todas las métricas

#### **analisis.py**
Análisis estadístico del dataset y reporte:
- `analizar_distribucion_datos()`: Estadísticas y balance
- `analizar_errores()`: Falsos positivos/negativos
- `generar_reporte_completo()`: Reporte JSON consolidado
- `imprimir_analisis_completo()`: Resumen legible en consola

#### **visualizaciones.py**
Genera gráficos en `resultados/`:
- Métricas, matriz de confusión, distribución de clases
- Palabras importantes y gráfico radar comparativo

#### **generar_dataset_grande.py**
Genera `datos_grande.csv` con más ejemplos balanceados.

#### **usar_modelo.py**
Carga `modelos/modelo_entrenado.pkl` y clasifica mensajes sin reentrenar.

#### **main.py**
Orquesta todo el pipeline:
1. Carga datos desde CSV
2. Divide en entrenamiento y prueba
3. Preprocesa los mensajes
4. Entrena el modelo
5. Evalúa con métricas
6. Muestra ejemplos de predicción
7. Analiza palabras importantes
8. Realiza análisis estadístico
9. Genera visualizaciones (si están disponibles)
10. Exporta reporte JSON y guarda el modelo

---

## **12. Explicación Detallada del Modelo Naive Bayes**

### **Fundamentos Matemáticos**

El algoritmo Naive Bayes está basado en el **Teorema de Bayes**:

\[
P(\text{clase} \mid \text{mensaje}) = \frac{P(\text{mensaje} \mid \text{clase}) \cdot P(\text{clase})}{P(\text{mensaje})}
\]

Para clasificación, solo necesitamos comparar probabilidades, por lo que podemos ignorar el denominador:

\[
P(\text{spam} \mid \text{mensaje}) \propto P(\text{mensaje} \mid \text{spam}) \cdot P(\text{spam})
\]

\[
P(\text{ham} \mid \text{mensaje}) \propto P(\text{mensaje} \mid \text{ham}) \cdot P(\text{ham})
\]

### **Suposición de Independencia (Naive)**

El modelo asume que las palabras son independientes entre sí (aunque esto no es completamente cierto en la realidad, funciona bien en la práctica). Esto nos permite calcular:

\[
P(\text{mensaje} \mid \text{clase}) = P(palabra_1 \mid \text{clase}) \cdot P(palabra_2 \mid \text{clase}) \cdot ... \cdot P(palabra_n \mid \text{clase})
\]

\[
P(\text{mensaje} \mid \text{clase}) = \prod_{i=1}^{n} P(palabra_i \mid \text{clase})
\]

### **Cálculo de Probabilidades**

#### **Probabilidades a Priori**

Se calculan simplemente como la proporción de mensajes de cada clase:

\[
P(\text{spam}) = \frac{\text{número de mensajes spam}}{\text{número total de mensajes}}
\]

\[
P(\text{ham}) = \frac{\text{número de mensajes ham}}{\text{número total de mensajes}}
\]

#### **Probabilidades Condicionales**

La probabilidad de una palabra dado una clase se calcula como:

\[
P(palabra \mid \text{clase}) = \frac{\text{frecuencia de palabra en clase}}{\text{total de palabras en clase}}
\]

#### **Suavizado de Laplace**

Para evitar problemas cuando una palabra no aparece en el entrenamiento de una clase (probabilidad = 0), usamos suavizado:

\[
P(palabra \mid \text{clase}) = \frac{\text{frecuencia} + \alpha}{\text{total palabras} + \alpha \cdot V}
\]

donde:
- \(\alpha = 1.0\) (parámetro de suavizado, comúnmente 1)
- \(V\) = tamaño del vocabulario (palabras únicas)

### **Uso de Logaritmos para Estabilidad Numérica**

Al multiplicar muchas probabilidades pequeñas, podemos tener problemas de **underflow** (números demasiado pequeños para representar). Por eso usamos logaritmos:

\[
\log(P(\text{clase} \mid \text{mensaje})) = \log(P(\text{clase})) + \sum_{i=1}^{n} \log(P(palabra_i \mid \text{clase}))
\]

Esto convierte multiplicaciones en sumas, que son más estables numéricamente.

### **Ejemplo Numérico Paso a Paso**

Supongamos que tenemos:

**Mensaje**: "dinero rápido"

**Datos de entrenamiento**:
- Spam: "dinero fácil dinero" (dinero aparece 2 veces, fácil 1 vez)
- Ham: "reunión mañana" (reunión y mañana aparecen 1 vez cada una)

**Vocabulario**: {dinero, fácil, reunión, mañana} → V = 4

**Paso 1: Probabilidades a priori**
- Total mensajes: 2
- P(spam) = 1/2 = 0.5
- P(ham) = 1/2 = 0.5

**Paso 2: Probabilidades condicionales con suavizado (α=1)**

Para spam:
- Total palabras en spam: 3
- P(dinero|spam) = (2 + 1) / (3 + 1×4) = 3/7 ≈ 0.429
- P(rápido|spam) = (0 + 1) / (3 + 1×4) = 1/7 ≈ 0.143

Para ham:
- Total palabras en ham: 2
- P(dinero|ham) = (0 + 1) / (2 + 1×4) = 1/6 ≈ 0.167
- P(rápido|ham) = (0 + 1) / (2 + 1×4) = 1/6 ≈ 0.167

**Paso 3: Calcular probabilidades finales**

Para spam:
- log(P(spam|mensaje)) = log(0.5) + log(0.429) + log(0.143) ≈ -0.693 - 0.846 - 1.946 ≈ -3.485
- P(spam|mensaje) ≈ exp(-3.485) ≈ 0.031

Para ham:
- log(P(ham|mensaje)) = log(0.5) + log(0.167) + log(0.167) ≈ -0.693 - 1.792 - 1.792 ≈ -4.277
- P(ham|mensaje) ≈ exp(-4.277) ≈ 0.014

**Paso 4: Normalizar y decidir**

- P(spam|mensaje) normalizada ≈ 0.031 / (0.031 + 0.014) ≈ 0.689
- P(ham|mensaje) normalizada ≈ 0.014 / (0.031 + 0.014) ≈ 0.311

**Resultado**: El mensaje se clasifica como **spam** (mayor probabilidad).

---

## **13. Guía Paso a Paso del Código**

### **Flujo de Ejecución del Script Principal (main.py)**

#### **Paso 1: Carga de Datos**

```python
import os
archivo = 'datos_grande.csv' if os.path.exists('datos_grande.csv') else 'datos.csv'
mensajes, etiquetas = cargar_datos(archivo)
```

**Qué hace:**
- Lee el archivo CSV línea por línea
- Extrae los campos `mensaje` y `etiqueta`
- Valida que las etiquetas sean 'spam' o 'ham'
- Retorna dos listas: una con mensajes y otra con etiquetas
 - Usa `datos_grande.csv` si existe, si no usa `datos.csv`

**Ejemplo de datos cargados:**
- mensajes = ["Gana dinero rápido", "Reunión confirmada", ...]
- etiquetas = ["spam", "ham", ...]

#### **Paso 2: División de Datos**

```python
mensajes_train, mensajes_test, etiquetas_train, etiquetas_test = dividir_datos(...)
```

**Qué hace:**
- Mezcla los datos aleatoriamente
- Separa el 80% para entrenamiento y 20% para prueba
- Esto permite evaluar el modelo con datos que no ha visto durante el entrenamiento

**¿Por qué es importante?**
- Evalúa si el modelo generaliza bien a datos nuevos
- Previene el sobreajuste (overfitting)

#### **Paso 3: Preprocesamiento**

```python
mensajes_train_preproc, vocab_train, _ = preprocesar_dataset(mensajes_train)
```

**Proceso interno:**

1. **Normalización** (`normalizar_texto`):
   - "¡Gana DINERO!" → "gana dinero"
   - Elimina puntuación y convierte a minúsculas

2. **Tokenización** (`tokenizar`):
   - "gana dinero" → ["gana", "dinero"]
   - Divide el texto en palabras

3. **Eliminación de stopwords** (`eliminar_stopwords`):
   - ["el", "dinero", "es", "fácil"] → ["dinero", "fácil"]
   - Remueve palabras comunes sin significado

4. **Creación de vocabulario**:
   - Recopila todas las palabras únicas de todos los mensajes
   - Asigna un índice único a cada palabra

**Resultado:** Lista de mensajes donde cada uno es una lista de tokens relevantes.

#### **Paso 4: Entrenamiento del Modelo**

```python
modelo = NaiveBayesSpamDetector(alpha=1.0)
modelo.entrenar(mensajes_train_preproc, etiquetas_train)
```

**Proceso interno de `entrenar()`:**

1. **Calcula probabilidades a priori:**
   ```python
   spam_count = contar mensajes con etiqueta 'spam'
   self.prob_spam = spam_count / total_mensajes
   ```

2. **Cuenta frecuencias de palabras:**
   ```python
   Para cada mensaje:
       Si es spam:
           Incrementar contador de palabras en spam_words
       Si es ham:
           Incrementar contador de palabras en ham_words
   ```

3. **Calcula probabilidades condicionales:**
   ```python
   Para cada palabra en vocabulario:
       P(palabra|spam) = (frecuencia_spam + alpha) / (total_spam + alpha * V)
       P(palabra|ham) = (frecuencia_ham + alpha) / (total_ham + alpha * V)
       Guardar log(P(palabra|clase)) para evitar underflow
   ```

**Resultado:** Modelo con todas las probabilidades aprendidas.

#### **Paso 5: Predicción**

```python
prediccion = modelo.predecir(mensaje_preproc)
```

**Proceso interno:**

1. **Preprocesa el mensaje** (si es necesario)

2. **Calcula log-probabilidades:**
   ```python
   log_P_spam = log(P(spam)) + sum(log(P(palabra|spam)) para cada palabra)
   log_P_ham = log(P(ham)) + sum(log(P(palabra|ham)) para cada palabra)
   ```

3. **Normaliza probabilidades:**
   ```python
   P_spam = exp(log_P_spam - max(log_P_spam, log_P_ham))
   P_ham = exp(log_P_ham - max(log_P_spam, log_P_ham))
   Normalizar para que sumen 1.0
   ```

4. **Retorna la clase con mayor probabilidad**

#### **Paso 6: Evaluación**

```python
resultados = evaluar_modelo(modelo, X_test, y_test)
```

**Proceso:**

1. **Hace predicciones** para todos los mensajes de prueba
2. **Calcula matriz de confusión:**
   - TP: Spam correctamente identificado
   - TN: Ham correctamente identificado
   - FP: Ham marcado como spam (error)
   - FN: Spam marcado como ham (error)

3. **Calcula métricas:**
   - Accuracy = (TP + TN) / Total
   - Precision = TP / (TP + FP)
   - Recall = TP / (TP + FN)
   - F1 = 2 × (Precision × Recall) / (Precision + Recall)

#### **Paso 7: Análisis y palabras importantes**

Se identifican palabras más características por clase y se generan estadísticas del dataset:
- Distribución y balance de clases
- Longitud promedio de mensajes
- Errores más comunes (FP/FN)

#### **Paso 8: Visualizaciones (opcional)**

Si están instaladas las dependencias, se generan gráficos en `resultados/`:
- Métricas, matriz de confusión, distribución de clases
- Palabras importantes y radar comparativo

#### **Paso 9: Reporte JSON y persistencia**

Se exporta un reporte completo en `resultados/reporte_completo.json` y se guarda el modelo en `modelos/modelo_entrenado.pkl`.

### **Explicación de Funciones Clave**

#### **Función `_calcular_log_probabilidad()` en modelo.py**

Esta función implementa el núcleo del Teorema de Bayes:

```python
def _calcular_log_probabilidad(self, mensaje, clase):
    # Empezar con log(P(clase))
    log_prob = self.log_prob_spam if clase == 'spam' else self.log_prob_ham
    
    # Sumar log(P(palabra|clase)) para cada palabra
    for palabra in mensaje:
        if palabra in prob_palabras:
            log_prob += prob_palabras[palabra]
        else:
            # Manejo de palabras desconocidas (OOV)
            log_prob += log(prob_oov)
    
    return log_prob
```

**¿Por qué logaritmos?**
- Multiplicar muchas probabilidades pequeñas puede causar underflow
- log(a × b) = log(a) + log(b), convertimos multiplicación en suma
- Más estable numéricamente

#### **Función `crear_bag_of_words()` en preprocesamiento.py**

Esta función convierte texto en números:

```python
def crear_bag_of_words(mensajes):
    # 1. Crear vocabulario: todas las palabras únicas
    vocabulario = {palabra: indice for indice, palabra in enumerate(palabras_unicas)}
    
    # 2. Para cada mensaje, crear un vector
    for mensaje in mensajes:
        vector = [0] * len(vocabulario)
        for palabra in mensaje:
            indice = vocabulario[palabra]
            vector[indice] += 1  # Incrementar contador
```

**Ejemplo:**
- Vocabulario: {"dinero": 0, "fácil": 1, "reunión": 2}
- Mensaje: "dinero fácil dinero"
- Vector: [2, 1, 0] (dinero aparece 2 veces, fácil 1 vez, reunión 0 veces)

---

## **14. Instrucciones de Uso**

### **Requisitos del Sistema**

- **Python 3.7 o superior**
- **Sistema operativo**: Windows, macOS o Linux
- **Bibliotecas**: El pipeline base funciona con bibliotecas estándar de Python. Para **visualizaciones** y análisis avanzados se recomienda instalar `matplotlib`, `seaborn`, `numpy`.

### **Instalación**

1. **Clonar o descargar el proyecto:**
   ```bash
   cd spam-detector-ai
   ```

2. **Verificar instalación de Python:**
   ```bash
   python --version
   # Debe mostrar Python 3.7 o superior
   ```

3. **Instalar dependencias:**

   **Opción A: Usar script automático (recomendado en macOS/Linux):**
   ```bash
   ./instalar_dependencias.sh
   source venv/bin/activate
   ```

   **Opción B: Instalación manual:**
   ```bash
   # Crear entorno virtual
   python3 -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   
   # Instalar dependencias
   pip install -r requirements.txt
   ```

   **Nota:** Si tienes problemas con el entorno gestionado de Python, el script automático creará un entorno virtual y instalará todo automáticamente.

### **Ejecución del Proyecto**

#### **Ejecutar el Pipeline Completo**

Simplemente ejecuta el script principal:

```bash
python main.py
```

Esto ejecutará:
1. Carga de datos
2. Preprocesamiento
3. Entrenamiento del modelo
4. Evaluación
5. Ejemplos de predicción
6. Análisis de palabras importantes
7. Análisis estadístico
8. Visualizaciones (si están disponibles)
9. Reporte JSON y guardado del modelo

**Nota:** El script usa `datos_grande.csv` si existe; de lo contrario usa `datos.csv`.
Para generar el dataset grande ejecuta:
```bash
python generar_dataset_grande.py
```

#### **Salida Esperada**

El programa mostrará:
- Progreso de cada paso
- Métricas de evaluación (accuracy, precision, recall, F1-score)
- Matriz de confusión
- Ejemplos de predicciones
- Palabras más características de spam y ham
- Análisis estadístico y recomendaciones
- Reporte JSON en `resultados/reporte_completo.json`
- Imágenes en `resultados/` (si están disponibles las dependencias)

### **Usar el Modelo para Clasificar Mensajes Propios**

Puedes cargar el modelo entrenado sin reentrenar:

```python
from preprocesamiento import preprocesar_mensaje
from modelo import NaiveBayesSpamDetector

# 1. Cargar el modelo guardado
modelo = NaiveBayesSpamDetector.cargar('modelos/modelo_entrenado.pkl')

# 2. Clasificar un nuevo mensaje
mensaje_nuevo = "Gana dinero rápido sin esfuerzo"
mensaje_preproc = preprocesar_mensaje(mensaje_nuevo)
prediccion = modelo.predecir(mensaje_preproc)
probabilidades = modelo.predecir_proba(mensaje_preproc)

print(f"Predicción: {prediccion}")
print(f"Probabilidades: Spam={probabilidades['spam']:.3f}, Ham={probabilidades['ham']:.3f}")
```

Primero ejecuta `python main.py` para generar el archivo `modelos/modelo_entrenado.pkl`.

También puedes usar el script interactivo:
```bash
python usar_modelo.py
```

### **Agregar Nuevos Datos**

Para agregar más ejemplos al dataset:

1. Abre `datos.csv` o `datos_grande.csv`
2. Agrega nuevas filas con el formato:
   ```csv
   id,mensaje,etiqueta
   181,"Tu nuevo mensaje aquí",spam
   ```
3. Asegúrate de que las etiquetas sean exactamente 'spam' o 'ham'
4. Ejecuta `main.py` nuevamente para reentrenar

### **Modificar Parámetros**

#### **Cambiar el porcentaje de datos de entrenamiento:**

En `main.py`, modifica:
```python
mensajes_train, mensajes_test, ... = dividir_datos(
    mensajes, etiquetas, 
    porcentaje_entrenamiento=0.7  # Cambiar de 0.8 a 0.7 (70% train, 30% test)
)
```

#### **Cambiar el parámetro de suavizado:**

En `main.py`, modifica:
```python
modelo = NaiveBayesSpamDetector(alpha=0.5)  # Cambiar de 1.0 a 0.5
```

Valores más altos de alpha dan más peso a palabras desconocidas.

### **Troubleshooting**

**Error: "FileNotFoundError: datos.csv"**
- Asegúrate de que `datos.csv` o `datos_grande.csv` estén en el mismo directorio que `main.py`
- Si falta el dataset grande, puedes generarlo con `python generar_dataset_grande.py`

**Error: "El modelo debe ser entrenado antes de hacer predicciones"**
- Ejecuta `modelo.entrenar()` antes de usar `modelo.predecir()`

**Advertencia: "Visualizaciones no disponibles"**
- Instala las dependencias con `pip install -r requirements.txt`

**Resultados muy bajos (accuracy < 0.7)**
- Revisa que el dataset esté balanceado (similar cantidad de spam y ham)
- Verifica que los mensajes estén correctamente etiquetados
- Considera agregar más datos de entrenamiento

---

## **15. Análisis de Resultados**

### **Interpretación de las Métricas**

#### **Accuracy (Exactitud)**

**Fórmula:** (TP + TN) / Total

**Interpretación:**
- **0.9 - 1.0**: Excelente - El modelo clasifica correctamente más del 90% de los casos
- **0.8 - 0.9**: Bueno - Desempeño sólido, pero hay margen de mejora
- **0.7 - 0.8**: Aceptable - Funciona, pero comete errores frecuentes
- **< 0.7**: Necesita mejoras - El modelo no está aprendiendo bien

**Limitación:** Puede ser engañosa si hay desbalance de clases. Si hay 95% ham y 5% spam, predecir siempre "ham" daría 95% accuracy sin aprender nada útil.

#### **Precision (Precisión)**

**Fórmula:** TP / (TP + FP)

**Interpretación:**
- **¿Qué significa?** Cuando el modelo dice "esto es spam", ¿qué tan a menudo tiene razón?
- **Alta precisión (> 0.9)**: Pocos falsos positivos - No marcamos correos legítimos como spam
- **Baja precisión (< 0.7)**: Muchos falsos positivos - Marcamos muchos correos legítimos como spam (malo para el usuario)

**Importancia:** En spam detection, la precisión es crítica porque marcar correos legítimos como spam es muy molesto para los usuarios.

#### **Recall (Sensibilidad)**

**Fórmula:** TP / (TP + FN)

**Interpretación:**
- **¿Qué significa?** De todo el spam que existe, ¿cuánto logramos capturar?
- **Alto recall (> 0.9)**: Capturamos casi todo el spam - Pocos falsos negativos
- **Bajo recall (< 0.7)**: Dejamos pasar mucho spam - Muchos falsos negativos (el spam llega a la bandeja de entrada)

**Importancia:** También es crítica porque dejar pasar spam es problemático.

#### **F1-Score**

**Fórmula:** 2 × (Precision × Recall) / (Precision + Recall)

**Interpretación:**
- Es un balance entre precisión y recall
- **Alto F1 (> 0.9)**: Buen balance entre capturar spam y no molestar a usuarios
- **Bajo F1 (< 0.7)**: Uno de los dos aspectos (precisión o recall) está fallando

**Ventaja:** Penaliza modelos que tienen una métrica muy alta y otra muy baja. Nos ayuda a encontrar el punto óptimo.

### **Matriz de Confusión**

La matriz de confusión nos muestra exactamente dónde está fallando el modelo:

```
                    Predicho
                  Spam    Ham
Realmente Spam     TP     FN
Realmente Ham      FP     TN
```

**Interpretación:**
- **TP alto, FN bajo**: Bien - Capturamos la mayoría del spam
- **TN alto, FP bajo**: Bien - No molestamos con falsas alarmas
- **FN alto**: Problema - Dejamos pasar mucho spam
- **FP alto**: Problema - Marcamos muchos correos legítimos como spam

### **Qué Significan los Valores Obtenidos**

#### **Escenario Ideal (Buen Modelo):**
```
Accuracy:  0.92 (92% correcto)
Precision: 0.90 (90% de los "spam" predichos son realmente spam)
Recall:    0.93 (93% del spam real es capturado)
F1-Score:  0.91 (balance entre precisión y recall)
```

#### **Escenario con Baja Precisión:**
```
Precision: 0.60
```
**Problema:** Muchos correos legítimos son marcados como spam  
**Solución:** Ajustar el umbral de decisión o mejorar el entrenamiento

#### **Escenario con Bajo Recall:**
```
Recall: 0.55
```
**Problema:** Mucho spam no está siendo detectado  
**Solución:** Agregar más ejemplos de spam al entrenamiento, ajustar suavizado

### **Análisis de Palabras Importantes**

El modelo también identifica las palabras más características de cada clase:

**Palabras típicas de SPAM:**
- dinero, gratis, rápido, gana, premio, click, millonario

**Palabras típicas de HAM:**
- reunión, confirmación, documento, informe, proyecto, gracias

Si estas palabras aparecen, el modelo está aprendiendo patrones correctos.

### **Cómo Mejorar el Modelo**

1. **Agregar más datos:**
   - Más ejemplos mejoran la generalización
   - Idealmente, tener miles de ejemplos

2. **Balancear el dataset:**
   - Similar cantidad de spam y ham
   - Si hay desbalance, el modelo puede sesgarse

3. **Ajustar el preprocesamiento:**
   - Modificar la lista de stopwords
   - Considerar lematización (agrupar variaciones: "ganar", "gana", "ganando")

4. **Ajustar el parámetro de suavizado:**
   - Probar diferentes valores de alpha (0.5, 1.0, 2.0)
   - Valores más altos dan más peso a palabras desconocidas

5. **Revisar y corregir etiquetas:**
   - Errores en las etiquetas del dataset afectan el aprendizaje
   - Validar manualmente algunos casos

6. **Considerar técnicas avanzadas:**
   - N-gramas (pares de palabras en lugar de palabras individuales)
   - TF-IDF en lugar de conteo simple
   - Otros algoritmos (SVM, Random Forest) para comparar

---

## **16. Resultados Esperados**

Al ejecutar el proyecto, se generan los siguientes resultados:

### **Métricas de Desempeño**

Con el dataset pequeño (`datos.csv`, 180 mensajes balanceados), se esperan resultados como:

| Métrica | Valor Esperado | Interpretación |
|---------|----------------|----------------|
| **Accuracy** | 0.85 - 0.95 | Excelente exactitud en la clasificación |
| **Precision** | 0.80 - 0.95 | Baja tasa de falsos positivos |
| **Recall** | 0.80 - 0.95 | Alta capacidad de detectar spam |
| **F1-Score** | 0.82 - 0.93 | Balance adecuado entre métricas |

### **Archivos Generados**

El script principal genera automáticamente:

1. **Visualizaciones** (en `resultados/`):
   - `metricas_desempeno.png` - Gráfico de barras con todas las métricas
   - `matriz_confusion.png` - Visualización de la matriz de confusión
   - `distribucion_clases.png` - Comparación de distribución de clases
   - `palabras_importantes.png` - Palabras clave por clase
   - `comparacion_metricas_radar.png` - Gráfico de radar comparativo
   - (Si no están instaladas las dependencias, estas visualizaciones se omiten)

2. **Reportes**:
   - `reporte_completo.json` - Reporte detallado en formato JSON con todos los análisis

### **Análisis Automático**

El sistema genera automáticamente:
- Análisis estadístico del dataset
- Identificación de errores (falsos positivos y negativos)
- Interpretación del desempeño
- Recomendaciones para mejorar el modelo

---

## **17. Estructura del Proyecto**

```
spam-detector-ai/
├── README.md                  # Documentación completa del proyecto
├── LICENSE                    # Licencia MIT
├── INSTALACION.md             # Guía de instalación
├── GUIA_MODELO_PERSISTENTE.md # Guía de persistencia del modelo
├── requirements.txt           # Dependencias del proyecto
├── .gitignore                 # Archivos a ignorar en Git
├── datos.csv                  # Dataset pequeño con 180 mensajes
├── datos_grande.csv           # Dataset grande con 1500 mensajes
├── generar_dataset_grande.py  # Generador del dataset grande
│
├── preprocesamiento.py        # Módulo de preprocesamiento de texto
├── modelo.py                  # Implementación de Naive Bayes
├── evaluacion.py              # Métricas de evaluación
├── visualizaciones.py         # Generación de gráficos profesionales
├── analisis.py                # Análisis estadístico detallado
├── usar_modelo.py             # Uso del modelo guardado
├── main.py                    # Script principal del pipeline
│
├── modelos/                   # Modelos guardados (se genera)
└── resultados/                # Directorio generado automáticamente
    ├── metricas_desempeno.png
    ├── matriz_confusion.png
    ├── distribucion_clases.png
    ├── palabras_importantes.png
    ├── comparacion_metricas_radar.png
    └── reporte_completo.json
```

---

## **18. Metodología de Desarrollo**

### **Fases del Proyecto**

1. **Análisis del Problema**
   - Identificación de la tarea de clasificación binaria
   - Selección del algoritmo apropiado (Naive Bayes)
   - Diseño de la arquitectura del sistema

2. **Preprocesamiento**
   - Normalización de texto
   - Tokenización y limpieza
   - Eliminación de ruido (stopwords, puntuación)

3. **Implementación del Modelo**
   - Desarrollo desde cero del algoritmo Naive Bayes
   - Implementación del suavizado de Laplace
   - Optimización para estabilidad numérica

4. **Evaluación**
   - División train/test
   - Cálculo de métricas estándar
   - Análisis de errores

5. **Visualización y Análisis**
   - Generación de gráficos profesionales
   - Análisis estadístico detallado
   - Exportación de reportes

### **Decisiones de Diseño**

- **Naive Bayes**: Elegido por su simplicidad, eficiencia y buen desempeño en NLP
- **Suavizado de Laplace**: Para manejar palabras no vistas en entrenamiento
- **Log-probabilidades**: Para evitar problemas de underflow numérico
- **Módulos separados**: Para facilitar mantenimiento y comprensión

---

## **19. Trabajos Futuros y Mejoras**

### **Mejoras Propuestas**

1. **Expansión del Dataset**
   - Aumentar a miles de ejemplos
   - Incluir más variabilidad en los mensajes
   - Datos de múltiples fuentes

2. **Técnicas Avanzadas**
   - Implementar n-gramas (bigramas, trigramas)
   - Usar TF-IDF en lugar de conteo simple
   - Considerar lematización y stemming

3. **Modelos Alternativos**
   - Comparar con SVM, Random Forest, Redes Neuronales
   - Ensambles de modelos
   - Modelos preentrenados (BERT, etc.)

4. **Optimización**
   - Optimización de hiperparámetros
   - Validación cruzada (k-fold)
   - Análisis de características más detallado

5. **Interfaz de Usuario**
   - Crear API REST
   - Desarrollar interfaz web
   - Aplicación móvil

---

## **20. Referencias y Bibliografía**

### **Referencias Académicas**

1. **Mitchell, T. M.** (1997). *Machine Learning*. McGraw-Hill.
   - Fundamentos de aprendizaje automático y clasificación

2. **Manning, C. D., Raghavan, P., & Schütze, H.** (2008). *Introduction to Information Retrieval*. Cambridge University Press.
   - Procesamiento de texto y clasificación de documentos

3. **Russell, S., & Norvig, P.** (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.
   - Algoritmos de IA y razonamiento probabilístico

### **Recursos Técnicos**

- **Scikit-learn Documentation**: https://scikit-learn.org/
  - Referencia para implementaciones de machine learning

- **NLTK Book**: https://www.nltk.org/book/
  - Procesamiento de lenguaje natural

- **Towards Data Science**: https://towardsdatascience.com/
  - Artículos sobre Naive Bayes y clasificación de texto

### **Documentación de Python**

- **Python Documentation**: https://docs.python.org/3/
- **Matplotlib Documentation**: https://matplotlib.org/
- **NumPy Documentation**: https://numpy.org/doc/

---

## **21. Agradecimientos**

Este proyecto fue desarrollado como trabajo final del curso de Inteligencia Artificial, implementando desde cero un sistema completo de clasificación de texto utilizando técnicas fundamentales de machine learning.

---

## **22. Información de Contacto y Licencia**

- **Licencia**: MIT License (ver archivo LICENSE)
- **Autor**: [Tu Nombre]
- **Institución**: [Nombre de la Universidad]
- **Curso**: Inteligencia Artificial
- **Año**: 2026

---

**Nota Final**: Este proyecto demuestra la implementación completa de un sistema de clasificación de texto desde cero, incluyendo preprocesamiento, modelado, evaluación y análisis. El código está diseñado para ser educativo, bien documentado y fácil de entender, ideal para propósitos académicos y de aprendizaje.

---

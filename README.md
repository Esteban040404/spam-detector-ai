# 📌 **Proyecto de IA — Algoritmo de Aprendizaje Supervisado**  
## ⭐ *Clasificación de Correos: ¿Spam o No Spam?*

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

Para este estudio se emplea un dataset pequeño, como el siguiente ejemplo:

| id | mensaje                                        | etiqueta |
|----|------------------------------------------------|----------|
| 1  | "Gana dinero rápido haciendo clic aquí"        | spam     |
| 2  | "Reunión confirmada para mañana"               | ham      |
| 3  | "Oferta limitada, compra ahora"                | spam     |
| 4  | "Adjunto envío los documentos solicitados"     | ham      |

El conjunto puede ampliarse para obtener mejores resultados, pero este tamaño permite mostrar el funcionamiento del modelo de forma clara.

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

Un ejemplo esperado con un dataset pequeño:

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

# Guía: Dataset Grande y Persistencia del Modelo

## 📊 Dataset Grande

El proyecto ahora incluye un generador de dataset grande que crea **1500 mensajes balanceados** (750 spam, 750 ham) con gran variedad y realismo.

### Generar el Dataset Grande

```bash
python3 generar_dataset_grande.py
```

Esto creará el archivo `datos_grande.csv` con 1500 mensajes variados.

### Ventajas del Dataset Grande

- **Mejor aprendizaje**: Más datos = mejor generalización
- **Mayor variedad**: 200+ plantillas diferentes con variaciones
- **Balanceado**: 50% spam, 50% ham
- **Realista**: Mensajes que simulan casos reales

## 💾 Persistencia del Modelo

El modelo ahora puede **guardar y cargar** su conocimiento aprendido, permitiendo:

1. **Reutilizar el modelo** sin reentrenar
2. **Aprendizaje incremental** con nuevos datos
3. **Preservar el conocimiento** entre sesiones

### Guardar el Modelo

El modelo se guarda automáticamente después del entrenamiento en:
```
modelos/modelo_entrenado.pkl
```

También puedes guardarlo manualmente:

```python
from modelo import NaiveBayesSpamDetector

# Entrenar modelo
modelo = NaiveBayesSpamDetector()
modelo.entrenar(X_train, y_train)

# Guardar modelo
modelo.guardar('modelos/mi_modelo.pkl')
```

### Cargar el Modelo

```python
from modelo import NaiveBayesSpamDetector

# Cargar modelo previamente entrenado
modelo = NaiveBayesSpamDetector.cargar('modelos/modelo_entrenado.pkl')

# Usar el modelo inmediatamente
prediccion = modelo.predecir("Gana dinero rápido")
print(prediccion)  # 'spam'
```

### Script de Uso Rápido

Usa el script incluido para cargar y usar el modelo:

```bash
python3 usar_modelo.py
```

Este script:
- Carga el modelo entrenado
- Muestra ejemplos de clasificación
- Permite modo interactivo para clasificar tus propios mensajes

## 🔄 Aprendizaje Incremental

El modelo puede **continuar aprendiendo** con nuevos datos sin perder el conocimiento previo:

```python
from modelo import NaiveBayesSpamDetector
from preprocesamiento import preprocesar_mensaje

# Cargar modelo existente
modelo = NaiveBayesSpamDetector.cargar('modelos/modelo_entrenado.pkl')

# Preparar nuevos datos
nuevos_mensajes = [
    preprocesar_mensaje("Nuevo mensaje spam"),
    preprocesar_mensaje("Nuevo mensaje ham")
]
nuevas_etiquetas = ['spam', 'ham']

# Continuar el entrenamiento
modelo.continuar_entrenamiento(nuevos_mensajes, nuevas_etiquetas)

# Guardar el modelo actualizado
modelo.guardar('modelos/modelo_entrenado.pkl')
```

### Ventajas del Aprendizaje Incremental

- **No pierde conocimiento previo**: Mantiene todo lo aprendido
- **Se adapta a nuevos patrones**: Aprende de nuevos ejemplos
- **Eficiente**: No necesita reentrenar desde cero
- **Útil para producción**: Puede mejorar con el tiempo

## 📈 Flujo de Trabajo Recomendado

### Primera Vez

1. **Generar dataset grande**:
   ```bash
   python3 generar_dataset_grande.py
   ```

2. **Entrenar modelo**:
   ```bash
   python3 main.py
   ```
   - Esto entrenará el modelo con el dataset grande
   - Guardará automáticamente el modelo en `modelos/modelo_entrenado.pkl`

### Uso Posterior

1. **Usar modelo existente**:
   ```bash
   python3 usar_modelo.py
   ```

2. **O cargar en tu código**:
   ```python
   from modelo import NaiveBayesSpamDetector
   modelo = NaiveBayesSpamDetector.cargar('modelos/modelo_entrenado.pkl')
   ```

### Mejorar el Modelo

1. **Agregar nuevos datos** al dataset
2. **Reentrenar** o usar **aprendizaje incremental**
3. **Evaluar** el nuevo desempeño
4. **Guardar** el modelo mejorado

## 🎯 Ejemplo Completo

```python
from modelo import NaiveBayesSpamDetector
from preprocesamiento import preprocesar_mensaje

# Opción 1: Cargar modelo existente
try:
    modelo = NaiveBayesSpamDetector.cargar('modelos/modelo_entrenado.pkl')
    print("✓ Modelo cargado desde archivo")
except FileNotFoundError:
    # Opción 2: Entrenar nuevo modelo
    from main import cargar_datos, dividir_datos
    from preprocesamiento import preprocesar_dataset
    
    mensajes, etiquetas = cargar_datos('datos_grande.csv')
    mensajes_train, mensajes_test, etiquetas_train, etiquetas_test = dividir_datos(
        mensajes, etiquetas, porcentaje_entrenamiento=0.8
    )
    mensajes_train_preproc, _, _ = preprocesar_dataset(mensajes_train)
    
    modelo = NaiveBayesSpamDetector()
    modelo.entrenar(mensajes_train_preproc, etiquetas_train)
    modelo.guardar('modelos/modelo_entrenado.pkl')
    print("✓ Modelo nuevo entrenado y guardado")

# Usar el modelo
mensaje = "Gana dinero rápido sin esfuerzo"
mensaje_preproc = preprocesar_mensaje(mensaje)
prediccion = modelo.predecir(mensaje_preproc)
probabilidades = modelo.predecir_proba(mensaje_preproc)

print(f"Mensaje: {mensaje}")
print(f"Predicción: {prediccion.upper()}")
print(f"Probabilidades: Spam={probabilidades['spam']:.3f}, Ham={probabilidades['ham']:.3f}")
```

## 📁 Estructura de Archivos

```
spam-detector-ai/
├── datos.csv              # Dataset pequeño original (180 mensajes)
├── datos_grande.csv       # Dataset grande generado (1500 mensajes)
├── generar_dataset_grande.py  # Script para generar dataset grande
├── usar_modelo.py         # Script para usar modelo entrenado
├── modelos/               # Directorio de modelos guardados
│   └── modelo_entrenado.pkl  # Modelo entrenado (se crea automáticamente)
└── ...
```

## ⚠️ Notas Importantes

1. **El modelo se guarda automáticamente** después de cada entrenamiento
2. **El archivo .pkl contiene todo el conocimiento** aprendido
3. **Puedes tener múltiples modelos** guardados con diferentes nombres
4. **El aprendizaje incremental** actualiza las probabilidades sin perder datos previos
5. **Los modelos son compatibles** entre diferentes ejecuciones

## 🚀 Mejoras Futuras

- Agregar versionado de modelos
- Implementar comparación de modelos
- Agregar métricas de desempeño al guardar
- Implementar validación automática al cargar

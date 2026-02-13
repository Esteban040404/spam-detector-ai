# Guía de Instalación y Configuración

## Requisitos Previos

- **Python 3.7 o superior**
- **pip** (gestor de paquetes de Python)
- **Sistema operativo**: Windows, macOS o Linux

## Instalación Paso a Paso

### 1. Verificar Instalación de Python

```bash
python --version
# Debe mostrar Python 3.7 o superior
```

Si no tienes Python instalado, descárgalo desde [python.org](https://www.python.org/downloads/)

### 2. Clonar o Descargar el Proyecto

Si tienes Git instalado:
```bash
git clone [URL_DEL_REPOSITORIO]
cd spam-detector-ai
```

O descarga el proyecto como ZIP y extráelo.

### 3. Crear Entorno Virtual (Recomendado)

**En Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**En macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Instalar Dependencias

```bash
pip install -r requirements.txt
```

Esto instalará:
- matplotlib (para visualizaciones)
- seaborn (para estilos de gráficos)
- numpy (para operaciones numéricas)

### 5. Verificar Instalación

```bash
python -c "import matplotlib; import numpy; print('✓ Todas las dependencias instaladas correctamente')"
```

## Ejecución del Proyecto

Una vez instaladas las dependencias, ejecuta:

```bash
python main.py
```

El script generará:
- Resultados en consola
- Gráficos en el directorio `resultados/`
- Reporte JSON con análisis completo

## Solución de Problemas

### Error: "No module named 'matplotlib'"

**Solución:** Instala las dependencias:
```bash
pip install -r requirements.txt
```

### Error: "FileNotFoundError: datos.csv"

**Solución:** Asegúrate de estar en el directorio correcto del proyecto:
```bash
cd spam-detector-ai
python main.py
```

### Error al generar gráficos

**Solución:** Si estás en un servidor sin interfaz gráfica, los gráficos se guardarán automáticamente. Si necesitas verlos, instala un backend de matplotlib:
```bash
pip install PyQt5
```

### Problemas con el entorno virtual

**Solución:** Si tienes problemas, puedes instalar las dependencias globalmente (no recomendado):
```bash
pip install matplotlib seaborn numpy
```

## Estructura de Directorios Después de la Ejecución

```
spam-detector-ai/
├── datos.csv
├── main.py
├── ... (otros archivos del proyecto)
└── resultados/          # Generado automáticamente
    ├── metricas_desempeno.png
    ├── matriz_confusion.png
    ├── distribucion_clases.png
    ├── palabras_importantes.png
    ├── comparacion_metricas_radar.png
    └── reporte_completo.json
```

## Próximos Pasos

Una vez instalado y ejecutado correctamente, puedes:
1. Modificar `datos.csv` para agregar más ejemplos
2. Ajustar parámetros en `main.py`
3. Explorar el código en los módulos individuales
4. Leer el `README.md` para documentación completa

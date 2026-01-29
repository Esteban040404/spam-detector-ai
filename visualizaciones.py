"""
Módulo de Visualización
========================

Este módulo contiene funciones para generar visualizaciones profesionales
de los resultados del modelo de clasificación de spam.

Incluye gráficos de:
- Métricas de desempeño
- Matriz de confusión
- Distribución de clases
- Palabras más importantes
- Análisis de errores
"""

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    MATPLOTLIB_DISPONIBLE = True
except ImportError:
    MATPLOTLIB_DISPONIBLE = False
    print("⚠ Advertencia: matplotlib no está instalado. Las visualizaciones no estarán disponibles.")
    print("   Instala con: pip install matplotlib seaborn numpy")

from collections import Counter


def configurar_estilo():
    """
    Configura el estilo de matplotlib para gráficos profesionales.
    """
    if not MATPLOTLIB_DISPONIBLE:
        raise ImportError("matplotlib no está instalado. Instala con: pip install matplotlib")
    
    # Intentar usar seaborn si está disponible, sino usar estilo por defecto
    try:
        import seaborn as sns
        sns.set_style("darkgrid")
    except ImportError:
        # Si seaborn no está disponible, usar estilo matplotlib
        plt.style.use('default')
    
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16


def graficar_metricas(resultados, ruta_guardado=None):
    """
    Genera un gráfico de barras con las métricas de desempeño del modelo.
    
    Parámetros:
    -----------
    resultados : dict
        Diccionario con las métricas de evaluación
    ruta_guardado : str, optional
        Ruta donde guardar el gráfico. Si es None, solo muestra el gráfico.
    """
    configurar_estilo()
    
    metricas = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    valores = [
        resultados['accuracy'],
        resultados['precision'],
        resultados['recall'],
        resultados['f1_score']
    ]
    
    # Crear colores degradados para las barras
    colores = plt.cm.viridis(np.linspace(0.3, 0.9, len(metricas)))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    barras = ax.bar(metricas, valores, color=colores, edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Agregar valores en las barras
    for i, (barra, valor) in enumerate(zip(barras, valores)):
        altura = barra.get_height()
        ax.text(barra.get_x() + barra.get_width()/2., altura + 0.01,
                f'{valor:.3f}\n({valor*100:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Configurar ejes
    ax.set_ylim([0, 1.1])
    ax.set_ylabel('Valor de la Métrica', fontweight='bold')
    ax.set_title('Métricas de Desempeño del Modelo Naive Bayes', fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Línea de referencia en 0.8 (buen desempeño)
    ax.axhline(y=0.8, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Umbral de Buen Desempeño (0.8)')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if ruta_guardado:
        plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico guardado en: {ruta_guardado}")
    
    plt.show()


def graficar_matriz_confusion(matriz, ruta_guardado=None):
    """
    Genera una visualización de la matriz de confusión con colores.
    
    Parámetros:
    -----------
    matriz : dict
        Diccionario con TP, TN, FP, FN
    ruta_guardado : str, optional
        Ruta donde guardar el gráfico.
    """
    configurar_estilo()
    
    # Crear matriz 2x2
    matriz_valores = np.array([
        [matriz['TP'], matriz['FN']],
        [matriz['FP'], matriz['TN']]
    ])
    
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Crear mapa de colores personalizado
    cmap = plt.cm.RdYlGn
    im = ax.imshow(matriz_valores, cmap=cmap, aspect='auto', vmin=0, vmax=matriz_valores.max())
    
    # Agregar texto en cada celda
    for i in range(2):
        for j in range(2):
            valor = matriz_valores[i, j]
            color = 'white' if valor > matriz_valores.max() / 2 else 'black'
            ax.text(j, i, f'{int(valor)}',
                   ha='center', va='center', color=color, fontsize=16, fontweight='bold')
    
    # Configurar etiquetas
    clases = ['Spam', 'Ham']
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicho\nSpam', 'Predicho\nHam'])
    ax.set_yticklabels(['Real\nSpam', 'Real\nHam'])
    
    # Agregar título
    ax.set_title('Matriz de Confusión', fontweight='bold', pad=20, fontsize=14)
    
    # Agregar barra de colores
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cantidad de Casos', rotation=270, labelpad=20, fontweight='bold')
    
    # Agregar anotaciones
    total = matriz_valores.sum()
    ax.text(0.5, -0.15, f'Total de casos: {int(total)}', 
            transform=ax.transAxes, ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    
    if ruta_guardado:
        plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
        print(f"✓ Matriz de confusión guardada en: {ruta_guardado}")
    
    plt.show()


def graficar_distribucion_clases(etiquetas_train, etiquetas_test, ruta_guardado=None):
    """
    Genera un gráfico comparando la distribución de clases en entrenamiento y prueba.
    
    Parámetros:
    -----------
    etiquetas_train : list
        Etiquetas del conjunto de entrenamiento
    etiquetas_test : list
        Etiquetas del conjunto de prueba
    ruta_guardado : str, optional
        Ruta donde guardar el gráfico.
    """
    configurar_estilo()
    
    # Contar clases
    train_spam = sum(1 for e in etiquetas_train if e == 'spam')
    train_ham = len(etiquetas_train) - train_spam
    test_spam = sum(1 for e in etiquetas_test if e == 'spam')
    test_ham = len(etiquetas_test) - test_spam
    
    # Preparar datos
    categorias = ['Spam', 'Ham']
    train_valores = [train_spam, train_ham]
    test_valores = [test_spam, test_ham]
    
    x = np.arange(len(categorias))
    ancho = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    barras1 = ax.bar(x - ancho/2, train_valores, ancho, label='Entrenamiento', 
                     color='#3498db', edgecolor='black', linewidth=1.5, alpha=0.8)
    barras2 = ax.bar(x + ancho/2, test_valores, ancho, label='Prueba', 
                     color='#e74c3c', edgecolor='black', linewidth=1.5, alpha=0.8)
    
    # Agregar valores en las barras
    for barras in [barras1, barras2]:
        for barra in barras:
            altura = barra.get_height()
            ax.text(barra.get_x() + barra.get_width()/2., altura,
                   f'{int(altura)}',
                   ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Clase', fontweight='bold')
    ax.set_ylabel('Cantidad de Mensajes', fontweight='bold')
    ax.set_title('Distribución de Clases en Conjuntos de Entrenamiento y Prueba', 
                 fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(categorias)
    ax.legend()
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if ruta_guardado:
        plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
        print(f"✓ Distribución de clases guardada en: {ruta_guardado}")
    
    plt.show()


def graficar_palabras_importantes(palabras_importantes, top_n=15, ruta_guardado=None):
    """
    Genera gráficos de barras horizontales con las palabras más importantes.
    
    Parámetros:
    -----------
    palabras_importantes : dict
        Diccionario con 'spam' y 'ham', cada uno con lista de tuplas (palabra, diferencia)
    top_n : int, default=15
        Número de palabras principales a mostrar
    ruta_guardado : str, optional
        Ruta donde guardar el gráfico.
    """
    configurar_estilo()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Gráfico para SPAM
    palabras_spam = palabras_importantes['spam'][:top_n]
    palabras_spam.reverse()  # Para mostrar la más importante arriba
    
    if palabras_spam:
        palabras = [p[0] for p in palabras_spam]
        diferencias = [p[1] for p in palabras_spam]
        
        ax1.barh(palabras, diferencias, color='#e74c3c', edgecolor='black', linewidth=1.2, alpha=0.8)
        ax1.set_xlabel('Diferencia de Probabilidad', fontweight='bold')
        ax1.set_title('Palabras Más Características de SPAM', fontweight='bold', pad=15)
        ax1.grid(axis='x', alpha=0.3, linestyle='--')
        ax1.set_axisbelow(True)
        
        # Agregar valores
        for i, (palabra, diff) in enumerate(zip(palabras, diferencias)):
            ax1.text(diff, i, f' {diff:.4f}', va='center', fontsize=9)
    
    # Gráfico para HAM
    palabras_ham = palabras_importantes['ham'][:top_n]
    palabras_ham.reverse()
    
    if palabras_ham:
        palabras = [p[0] for p in palabras_ham]
        diferencias = [p[1] for p in palabras_ham]
        
        ax2.barh(palabras, diferencias, color='#3498db', edgecolor='black', linewidth=1.2, alpha=0.8)
        ax2.set_xlabel('Diferencia de Probabilidad', fontweight='bold')
        ax2.set_title('Palabras Más Características de HAM', fontweight='bold', pad=15)
        ax2.grid(axis='x', alpha=0.3, linestyle='--')
        ax2.set_axisbelow(True)
        
        # Agregar valores
        for i, (palabra, diff) in enumerate(zip(palabras, diferencias)):
            ax2.text(diff, i, f' {diff:.4f}', va='center', fontsize=9)
    
    plt.suptitle('Análisis de Palabras Clave por Clase', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if ruta_guardado:
        plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
        print(f"✓ Palabras importantes guardadas en: {ruta_guardado}")
    
    plt.show()


def graficar_comparacion_metricas(resultados, ruta_guardado=None):
    """
    Genera un gráfico de radar (spider chart) comparando todas las métricas.
    
    Parámetros:
    -----------
    resultados : dict
        Diccionario con las métricas
    ruta_guardado : str, optional
        Ruta donde guardar el gráfico.
    """
    configurar_estilo()
    
    # Preparar datos
    metricas = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    valores = [
        resultados['accuracy'],
        resultados['precision'],
        resultados['recall'],
        resultados['f1_score']
    ]
    
    # Ángulos para el gráfico de radar
    angulos = np.linspace(0, 2 * np.pi, len(metricas), endpoint=False).tolist()
    valores += valores[:1]  # Cerrar el círculo
    angulos += angulos[:1]
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Dibujar el gráfico
    ax.plot(angulos, valores, 'o-', linewidth=2, color='#3498db', label='Desempeño del Modelo')
    ax.fill(angulos, valores, alpha=0.25, color='#3498db')
    
    # Configurar etiquetas
    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels(metricas, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Agregar valores en los puntos
    for angulo, valor, metrica in zip(angulos[:-1], valores[:-1], metricas):
        ax.text(angulo, valor + 0.05, f'{valor:.3f}', 
               ha='center', va='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title('Comparación de Métricas de Desempeño\n(Gráfico de Radar)', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if ruta_guardado:
        plt.savefig(ruta_guardado, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico de radar guardado en: {ruta_guardado}")
    
    plt.show()


def generar_reporte_visual(resultados, etiquetas_train, etiquetas_test, 
                          palabras_importantes, directorio='resultados'):
    """
    Genera todos los gráficos y los guarda en un directorio.
    
    Parámetros:
    -----------
    resultados : dict
        Resultados de la evaluación
    etiquetas_train : list
        Etiquetas de entrenamiento
    etiquetas_test : list
        Etiquetas de prueba
    palabras_importantes : dict
        Palabras importantes por clase
    directorio : str, default='resultados'
        Directorio donde guardar los gráficos
    """
    import os
    
    # Crear directorio si no existe
    os.makedirs(directorio, exist_ok=True)
    
    print(f"\n📊 Generando visualizaciones y guardándolas en '{directorio}/'...\n")
    
    # Generar todos los gráficos
    graficar_metricas(resultados, os.path.join(directorio, 'metricas_desempeno.png'))
    graficar_matriz_confusion(resultados['matriz_confusion'], 
                             os.path.join(directorio, 'matriz_confusion.png'))
    graficar_distribucion_clases(etiquetas_train, etiquetas_test,
                                os.path.join(directorio, 'distribucion_clases.png'))
    graficar_palabras_importantes(palabras_importantes, top_n=15,
                                 ruta_guardado=os.path.join(directorio, 'palabras_importantes.png'))
    graficar_comparacion_metricas(resultados,
                                  os.path.join(directorio, 'comparacion_metricas_radar.png'))
    
    print(f"\n✓ Todas las visualizaciones han sido guardadas en '{directorio}/'")

"""
Script para Usar el Modelo Entrenado
====================================

Este script muestra cómo cargar y usar un modelo previamente entrenado
sin necesidad de reentrenar desde cero.
"""

from modelo import NaiveBayesSpamDetector
from preprocesamiento import preprocesar_mensaje
import os


def usar_modelo_entrenado():
    """
    Carga y usa un modelo previamente entrenado.
    """
    ruta_modelo = 'modelos/modelo_entrenado.pkl'
    
    if not os.path.exists(ruta_modelo):
        print(f"❌ Error: No se encontró el modelo en {ruta_modelo}")
        print("   Primero debes entrenar el modelo ejecutando: python main.py")
        return
    
    # Cargar el modelo
    print("="*60)
    print("CARGANDO MODELO ENTRENADO")
    print("="*60)
    modelo = NaiveBayesSpamDetector.cargar(ruta_modelo)
    
    print("\n" + "="*60)
    print("CLASIFICACIÓN DE MENSAJES")
    print("="*60)
    
    # Ejemplos de mensajes para clasificar
    mensajes_ejemplo = [
        "Gana dinero rápido sin esfuerzo haciendo clic aquí",
        "Reunión confirmada para mañana a las 3pm en la sala de juntas",
        "Has ganado un iPhone gratis envía tus datos ahora",
        "El informe mensual está listo para revisión",
        "Urgente llama ya y gana un viaje gratis",
        "Gracias por tu correo te responderé pronto",
        "Millonario en 30 días garantizado sin inversión",
        "Confirmación de tu cita médica el próximo martes",
        "Click aquí para reclamar tu herencia millonaria",
        "Los documentos fueron enviados correctamente"
    ]
    
    print("\nClasificando mensajes de ejemplo:\n")
    for i, mensaje in enumerate(mensajes_ejemplo, 1):
        mensaje_preproc = preprocesar_mensaje(mensaje)
        prediccion = modelo.predecir(mensaje_preproc)
        probabilidades = modelo.predecir_proba(mensaje_preproc)
        
        # Determinar confianza
        confianza = max(probabilidades.values())
        nivel_confianza = "Alta" if confianza > 0.8 else "Media" if confianza > 0.6 else "Baja"
        
        print(f"{i}. Mensaje: {mensaje[:60]}...")
        print(f"   Predicción: {prediccion.upper()} (Confianza: {nivel_confianza})")
        print(f"   Probabilidades: Spam={probabilidades['spam']:.3f}, Ham={probabilidades['ham']:.3f}")
        print()
    
    # Modo interactivo
    print("="*60)
    print("MODO INTERACTIVO")
    print("="*60)
    print("\nEscribe mensajes para clasificar (escribe 'salir' para terminar):\n")
    
    while True:
        mensaje_usuario = input("Mensaje: ").strip()
        
        if mensaje_usuario.lower() in ['salir', 'exit', 'quit', 'q']:
            print("\n¡Hasta luego!")
            break
        
        if not mensaje_usuario:
            continue
        
        mensaje_preproc = preprocesar_mensaje(mensaje_usuario)
        prediccion = modelo.predecir(mensaje_preproc)
        probabilidades = modelo.predecir_proba(mensaje_preproc)
        
        confianza = max(probabilidades.values())
        nivel_confianza = "Alta" if confianza > 0.8 else "Media" if confianza > 0.6 else "Baja"
        
        print(f"\n  → Predicción: {prediccion.upper()} (Confianza: {nivel_confianza})")
        print(f"    Spam: {probabilidades['spam']:.1%} | Ham: {probabilidades['ham']:.1%}\n")


if __name__ == "__main__":
    usar_modelo_entrenado()

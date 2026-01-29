"""
Script para Generar Dataset Grande
===================================

Este script genera un dataset grande y variado de mensajes spam y ham
para entrenar el modelo con más datos y mejorar su desempeño.
"""

import csv
import random

# Plantillas de mensajes SPAM
plantillas_spam = [
    # Ofertas y promociones
    "Gana dinero rápido sin esfuerzo",
    "Oferta exclusiva solo por tiempo limitado",
    "Millones de pesos esperando por ti",
    "Gratis gana dinero desde tu casa",
    "Click aquí para reclamar tu premio",
    "Has ganado un iPhone gratis",
    "Dinero fácil trabajando desde casa",
    "Urgente llama ya y gana un viaje",
    "Premio garantizado para nuevos miembros",
    "Gana millones con un solo click",
    "Sin costo gana premios increíbles",
    "Millonario en 30 días garantizado",
    "Regístrate gratis y gana dinero",
    "Oferta limitada compra ahora",
    "Has sido seleccionado para ganar",
    "Click aquí para ser millonario",
    "Dinero rápido y seguro sin riesgo",
    "Gratis millones solo por registrarte",
    "Urgente reclamar premio antes de hoy",
    "Millones fáciles trabajando desde casa",
    
    # Loterías y sorteos
    "Has ganado la lotería internacional",
    "Felicidades eres el ganador del sorteo",
    "Reclama tu premio millonario ahora",
    "Has sido elegido ganador del concurso",
    "Tu número fue seleccionado para el premio",
    "Ganaste un auto nuevo completamente gratis",
    "Premio sorpresa esperando por ti",
    "Has ganado un viaje todo pagado",
    "Tu herencia millonaria te está esperando",
    "Reclama tu herencia antes que expire",
    
    # Trabajos desde casa
    "Trabaja desde casa gana dinero fácil",
    "Empleo desde casa sin experiencia necesaria",
    "Gana dinero mientras duermes",
    "Trabajo online desde cualquier lugar",
    "Ingresos pasivos desde tu computadora",
    "Gana dinero con tu teléfono móvil",
    "Trabajo remoto sin horarios fijos",
    "Empleo desde casa sin salir",
    "Gana dinero fácil sin conocimientos",
    "Trabajo desde casa garantizado",
    
    # Productos y servicios
    "Píldoras milagrosas para bajar de peso",
    "Producto exclusivo no disponible en tiendas",
    "Oferta especial solo para ti",
    "Descuento increíble solo hoy",
    "Compra ahora y ahorra millones",
    "Producto revolucionario que cambiará tu vida",
    "Oferta única no te la pierdas",
    "Precio especial por tiempo limitado",
    "Producto exclusivo para clientes VIP",
    "Oferta especial solo esta semana",
    
    # Inversiones y criptomonedas
    "Invierte ahora y gana millones",
    "Bitcoin te hará millonario",
    "Inversión segura con retorno garantizado",
    "Gana dinero con criptomonedas fácil",
    "Inversión sin riesgo con altos rendimientos",
    "Multiplica tu dinero en días",
    "Oportunidad de inversión única",
    "Invierte poco y gana mucho",
    "Inversión garantizada sin pérdidas",
    "Gana dinero con trading automático",
    
    # Salud y belleza
    "Píldoras para perder peso sin dieta",
    "Producto milagroso para el cabello",
    "Crema anti edad resultados inmediatos",
    "Suplemento que te hará más joven",
    "Producto exclusivo no disponible en farmacias",
    "Cura milagrosa para todas las enfermedades",
    "Producto natural sin efectos secundarios",
    "Resultados garantizados en una semana",
    "Producto revolucionario de belleza",
    "Transforma tu cuerpo sin esfuerzo",
    
    # Educación y cursos
    "Curso que te hará millonario",
    "Aprende a ganar dinero en línea",
    "Curso exclusivo solo por tiempo limitado",
    "Certificación que cambiará tu vida",
    "Diploma internacional sin estudiar",
    "Curso online con garantía de empleo",
    "Aprende habilidades que te harán rico",
    "Curso premium con descuento especial",
    "Educación que te hará exitoso",
    "Capacitación que multiplicará tus ingresos",
    
    # Préstamos y créditos
    "Préstamo rápido sin papeleos",
    "Crédito aprobado sin verificación",
    "Préstamo sin intereses por tiempo limitado",
    "Dinero rápido sin garantías",
    "Préstamo instantáneo sin buro de crédito",
    "Crédito fácil sin requisitos complicados",
    "Préstamo urgente en minutos",
    "Dinero inmediato sin esperas",
    "Préstamo sin aval ni fiador",
    "Crédito garantizado para todos",
    
    # Más variaciones
    "Urgente acción requerida reclamar ahora",
    "Oferta expira en las próximas horas",
    "Solo hoy precio especial",
    "Última oportunidad no la pierdas",
    "Exclusivo para miembros seleccionados",
    "Oferta secreta solo para ti",
    "Gana dinero infinito con este método",
    "Millonario en días no meses",
    "Dinero fácil garantizado sin esfuerzo",
    "Oportunidad única de por vida"
]

# Plantillas de mensajes HAM (legítimos)
plantillas_ham = [
    # Reuniones y citas
    "Reunión confirmada para mañana a las 3pm",
    "Confirmación de tu cita médica el próximo martes",
    "La reunión de equipo será el viernes",
    "Recordatorio reunión de seguimiento mañana",
    "Confirmación de asistencia al seminario",
    "La junta directiva será el próximo lunes",
    "Reunión de coordinación el miércoles",
    "Recordatorio reunión de planeación",
    "Confirmación de tu reserva para el evento",
    "La reunión de seguridad será mañana",
    
    # Trabajo y proyectos
    "El informe mensual está listo para revisión",
    "Los resultados del proyecto fueron excelentes",
    "La propuesta está lista para su aprobación",
    "El informe trimestral está disponible",
    "Los documentos están listos para firma",
    "El presupuesto fue aprobado por la dirección",
    "La evaluación del proyecto fue exitosa",
    "El informe de ventas mensual está listo",
    "Los resultados superaron las expectativas",
    "El análisis financiero fue completado",
    
    # Comunicación profesional
    "Gracias por tu correo te responderé pronto",
    "Te envío la presentación que solicitaste",
    "Adjunto envío los documentos solicitados",
    "Gracias por tu colaboración en el proyecto",
    "Te agradezco por tu pronta respuesta",
    "Confirmación de tu solicitud procesada",
    "Los documentos fueron enviados correctamente",
    "Gracias por tu dedicación al trabajo",
    "Te envío la información solicitada",
    "Confirmación de entrega del pedido",
    
    # Informes y reportes
    "El informe de calidad está disponible",
    "Los reportes mensuales fueron generados",
    "El análisis de mercado está completo",
    "El informe de auditoría está disponible",
    "Los informes financieros están completos",
    "El informe anual está disponible para consulta",
    "El informe de ventas trimestral está listo",
    "Los resultados fueron positivos en todos los aspectos",
    "El informe de desempeño fue exitoso",
    "El análisis de costos está completo",
    
    # Confirmaciones y registros
    "Confirmación de tu registro en el sistema",
    "Confirmación de tu suscripción al servicio",
    "Confirmación de tu membresía activa",
    "Confirmación de tu inscripción al curso",
    "Confirmación de tu alta en el sistema",
    "Confirmación de tu solicitud de vacaciones",
    "Confirmación de tu registro exitoso",
    "Confirmación de tu participación en el evento",
    "Confirmación de asistencia confirmada",
    "Confirmación de entrega de materiales",
    
    # Recordatorios
    "Recordatorio pago de nómina el día 15",
    "Recordatorio pago de servicios",
    "Recordatorio pago de facturas pendientes",
    "Recordatorio evaluación de desempeño",
    "Recordatorio capacitación del personal",
    "Recordatorio capacitación de nuevos empleados",
    "Recordatorio capacitación de seguridad",
    "Recordatorio junta directiva el viernes",
    "Recordatorio junta de trabajo mañana",
    "Recordatorio reunión de seguimiento",
    
    # Actualizaciones y notificaciones
    "El presupuesto del proyecto fue actualizado",
    "Los documentos legales fueron revisados",
    "El contrato fue firmado exitosamente",
    "El presupuesto operativo fue aprobado",
    "Los documentos fueron firmados correctamente",
    "El contrato de servicios fue renovado",
    "El presupuesto de marketing fue aprobado",
    "El contrato fue modificado según lo acordado",
    "Los documentos legales fueron actualizados",
    "El presupuesto del departamento fue actualizado",
    
    # Agradecimientos
    "Gracias por tu excelente trabajo",
    "Te agradezco por tu profesionalismo",
    "Gracias por tu compromiso con la empresa",
    "Gracias por tu contribución al equipo",
    "Gracias por tu puntualidad en las entregas",
    "Gracias por tu trabajo excepcional",
    "Te agradezco por tu trabajo excepcional",
    "Gracias por tu participación activa",
    "Te agradezco por tu dedicación",
    "Gracias por tu colaboración continua",
    
    # Más variaciones profesionales
    "El análisis de resultados fue concluyente",
    "Los resultados fueron excelentes en todos los rubros",
    "La propuesta técnica fue aceptada completamente",
    "La propuesta comercial fue aceptada",
    "La propuesta de mejoras fue bien recibida",
    "El seguimiento del proyecto está en curso",
    "Los documentos están listos para revisión",
    "El proceso de aprobación fue completado",
    "La documentación fue actualizada correctamente",
    "El sistema fue configurado exitosamente",
    
    # Comunicaciones formales
    "Por favor revisa el documento adjunto",
    "Te envío los archivos que solicitaste",
    "Los materiales están disponibles en el sistema",
    "La información fue procesada correctamente",
    "El procedimiento fue completado sin inconvenientes",
    "La solicitud fue recibida y está en proceso",
    "El trámite fue realizado exitosamente",
    "La gestión fue completada según lo programado",
    "El proceso fue finalizado correctamente",
    "La operación fue ejecutada sin errores"
]


def generar_variaciones(mensaje_base, num_variaciones=3):
    """
    Genera variaciones de un mensaje base agregando palabras o modificando ligeramente.
    """
    variaciones = [mensaje_base]  # Incluir el mensaje original
    
    palabras = mensaje_base.split()
    
    for _ in range(num_variaciones - 1):
        # Crear variación agregando o modificando palabras
        variacion = list(palabras)
        
        # Agregar palabra al inicio o final ocasionalmente
        if random.random() < 0.3:
            palabras_extra = ["importante", "urgente", "confirmado", "aprobado", "listo"]
            variacion.insert(0, random.choice(palabras_extra))
        
        # Ocasionalmente cambiar una palabra
        if random.random() < 0.4 and len(variacion) > 2:
            idx = random.randint(0, len(variacion) - 1)
            sinónimos = {
                "mañana": ["próximo día", "el día siguiente"],
                "reunión": ["junta", "sesión", "encuentro"],
                "documento": ["archivo", "informe", "reporte"],
                "confirmado": ["aprobado", "verificado", "validado"],
                "listo": ["completo", "finalizado", "terminado"]
            }
            palabra_actual = variacion[idx].lower()
            if palabra_actual in sinónimos:
                variacion[idx] = random.choice(sinónimos[palabra_actual])
        
        variaciones.append(" ".join(variacion))
    
    return variaciones


def generar_dataset_grande(nombre_archivo='datos_grande.csv', total_mensajes=1500):
    """
    Genera un dataset grande con mensajes variados.
    
    Parámetros:
    -----------
    nombre_archivo : str
        Nombre del archivo CSV a generar
    total_mensajes : int
        Número total de mensajes a generar (aproximadamente)
    """
    mensajes = []
    id_counter = 1
    
    # Calcular cuántos mensajes de cada tipo
    num_spam = total_mensajes // 2
    num_ham = total_mensajes - num_spam
    
    print(f"Generando dataset con {total_mensajes} mensajes...")
    print(f"  - Spam: {num_spam} mensajes")
    print(f"  - Ham: {num_ham} mensajes")
    
    # Generar mensajes SPAM
    print("\nGenerando mensajes SPAM...")
    spam_generados = 0
    while spam_generados < num_spam:
        # Seleccionar plantilla aleatoria
        plantilla = random.choice(plantillas_spam)
        
        # Generar variaciones
        variaciones = generar_variaciones(plantilla, num_variaciones=random.randint(2, 4))
        
        for variacion in variaciones:
            if spam_generados >= num_spam:
                break
            mensajes.append({
                'id': id_counter,
                'mensaje': variacion,
                'etiqueta': 'spam'
            })
            id_counter += 1
            spam_generados += 1
            
            if spam_generados % 100 == 0:
                print(f"  Generados {spam_generados}/{num_spam} mensajes spam...")
    
    # Generar mensajes HAM
    print("\nGenerando mensajes HAM...")
    ham_generados = 0
    while ham_generados < num_ham:
        # Seleccionar plantilla aleatoria
        plantilla = random.choice(plantillas_ham)
        
        # Generar variaciones
        variaciones = generar_variaciones(plantilla, num_variaciones=random.randint(2, 4))
        
        for variacion in variaciones:
            if ham_generados >= num_ham:
                break
            mensajes.append({
                'id': id_counter,
                'mensaje': variacion,
                'etiqueta': 'ham'
            })
            id_counter += 1
            ham_generados += 1
            
            if ham_generados % 100 == 0:
                print(f"  Generados {ham_generados}/{num_ham} mensajes ham...")
    
    # Mezclar los mensajes para que no estén agrupados por tipo
    random.shuffle(mensajes)
    
    # Escribir al archivo CSV
    print(f"\nEscribiendo dataset a {nombre_archivo}...")
    with open(nombre_archivo, 'w', encoding='utf-8', newline='') as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=['id', 'mensaje', 'etiqueta'])
        escritor.writeheader()
        escritor.writerows(mensajes)
    
    print(f"\n✓ Dataset generado exitosamente!")
    print(f"  - Total de mensajes: {len(mensajes)}")
    print(f"  - Spam: {spam_generados}")
    print(f"  - Ham: {ham_generados}")
    print(f"  - Archivo: {nombre_archivo}")


if __name__ == "__main__":
    # Generar dataset grande (1500 mensajes)
    generar_dataset_grande('datos_grande.csv', total_mensajes=1500)
    
    print("\n" + "="*60)
    print("Dataset grande generado. Ahora puedes usarlo ejecutando:")
    print("  python main.py")
    print("="*60)

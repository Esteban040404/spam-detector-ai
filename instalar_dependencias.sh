#!/bin/bash

# Script para instalar dependencias del proyecto
# Este script crea un entorno virtual e instala las dependencias necesarias

echo "=========================================="
echo "Instalación de Dependencias"
echo "=========================================="
echo ""

# Verificar si ya existe un entorno virtual
if [ -d "venv" ]; then
    echo "⚠ Entorno virtual ya existe. ¿Deseas recrearlo? (s/n)"
    read -r respuesta
    if [ "$respuesta" = "s" ]; then
        echo "Eliminando entorno virtual existente..."
        rm -rf venv
    else
        echo "Usando entorno virtual existente."
        source venv/bin/activate
        pip install -r requirements.txt
        echo ""
        echo "✓ Dependencias instaladas"
        echo ""
        echo "Para activar el entorno virtual en el futuro:"
        echo "  source venv/bin/activate"
        exit 0
    fi
fi

# Crear entorno virtual
echo "Creando entorno virtual..."
python3 -m venv venv

# Activar entorno virtual
echo "Activando entorno virtual..."
source venv/bin/activate

# Actualizar pip
echo "Actualizando pip..."
pip install --upgrade pip

# Instalar dependencias
echo "Instalando dependencias..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "✓ Instalación completada exitosamente"
echo "=========================================="
echo ""
echo "Para usar el proyecto:"
echo "  1. Activa el entorno virtual:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Ejecuta el proyecto:"
echo "     python3 main.py"
echo ""
echo "Para desactivar el entorno virtual:"
echo "     deactivate"
echo ""

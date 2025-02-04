#!/bin/bash

# Guarda la ruta actual
CURRENT_DIR=$(pwd)

# Ruta relativa al entorno virtual desde el directorio actual
VENV_PATH="../ia"

# Verificar si el directorio del entorno virtual existe
if [ -d "$VENV_PATH" ]; then
  # Cambiar al directorio del entorno virtual
  cd "$VENV_PATH" || { echo "No se pudo cambiar al directorio $VENV_PATH"; exit 1; }
  
  # Verificar si el script de activación existe
  if [ -f "bin/activate" ]; then
    # Activar el entorno virtual
    source bin/activate
    echo "Entorno virtual activado."
  else
    echo "El script de activación no existe en $VENV_PATH/bin/activate"
    exit 1
  fi
  
  # Regresar al directorio original
  cd "$CURRENT_DIR" || { echo "No se pudo regresar al directorio $CURRENT_DIR"; exit 1; }
else
  echo "El entorno virtual no existe en $VENV_PATH"
  exit 1
fi

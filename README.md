# Ejercicios de Programación

Este repositorio contiene soluciones a diversos ejercicios de programación, con los enunciados disponibles en formato PNG en la carpeta `exercises`. Los archivos de código para cada ejercicio están nombrados siguiendo la convención **P**ractica-**E**jercicio (Ejemplo: `P2E2.py` para la Parte 2 de la Práctica 2). A continuación, se detalla la estructura del repositorio y la funcionalidad de cada archivo.

## Estructura del repositorio

- **exercises/**: Carpeta que contiene los enunciados de los ejercicios en formato PNG.
- **P2E2.py**: Solución a la Parte 2 de la Práctica 2.
- **P2E3_trained_lambert.py**: Archivo que contiene la solución para el Ejercicio 3 usando una implementación de la función `lambert` entrenada manualmente.
- **P2E3_scipy_lambert.py**: Archivo que contiene la solución para el Ejercicio 3 usando `lambertw` de la librería `scipy.special`.
- **requirements.txt**: Requerimientos usados por los archivos
- **main.tex**: Memoria de la práctica en formato LaTeX
- **lambert_model.pkl**: Modelo entrenado en `P2E2.py` para la función de lambert. Contiene los pesos y bias de la red.
- **VD_model.pkl**: Modelo entrenado en `P2E3_trained_lambert.py` para el cálculo de la tensión del diodo. Contiene los pesos y bias de la red.

## Ejercicios

### Ejercicio 2
- **Archivo**: `P2E2.py`
- **Descripción**: Entrena un modelo para predecir valores de la función de lambert y lo guarda en `lambert_model.pkl`

### Ejercicio 3
- **Archivos**:
  - `P2E3_trained_lambert.py`: Entrena un modelo para obtener la tension del diodo utilizando una versión entrenada de la función `lambert` (`lambert_model.pkl`) en `P2E2.py`, y guarda el modelo en `VD_model.pkl`.
  - `P2E3_scipy_lambert.py`: Resuelve el mismo ejercicio, pero utilizando `lambertw` de `scipy.special`.
- **Descripción**: Ambos archivos implementan la misma funcionalidad usando métodos diferentes. Se realiza la comparativa dada la poca fiabilidad de la función lambert entrenada al ser solo un epoch de entrenamiento y usando 3 muestras de entrada.

## Requisitos
Algunos archivos pueden requerir bibliotecas adicionales como `scipy`, `numpy` y `matplotlib`. Puedes instalarlas ejecutando:

```bash
pip install -r requirements.txt
```

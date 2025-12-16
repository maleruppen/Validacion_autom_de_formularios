
# Trabajo Práctico 1 - Procesamiento de Imágenes

Este repositorio contiene las soluciones para los dos ejercicios del Trabajo Práctico N° 1 de Procesamiento de Imágenes.

## 📄 Documentación
* **[Consigna del Trabajo](Informe_TPPDI.pdf)**
* **[Informe Final](Informe_TPPDI.pdf)**

## 📂 Estructura del Proyecto
El proyecto está organizado de la siguiente manera:
* **`/src`**: Contiene los códigos fuente (`ej1.py`, `ej2.py`).
* **`/data`**: Contiene las imágenes de entrada (formularios y .tif).
* **`/`**: Archivos PDF y documentación.

---

## 📋 Prerrequisitos

* **Python 3**: Asegúrate de tener Python 3 instalado en tu sistema. Puedes descargarlo desde [python.org](https://www.python.org/).

---

## ⚙️ Configuración del Entorno

Se recomienda utilizar un entorno virtual para gestionar las dependencias del proyecto.

1.  **Crear el entorno virtual:**
    Abre una terminal o línea de comandos en la carpeta del proyecto y ejecuta:
    ```bash
    python -m venv .venv
    ```
    *(Reemplaza `.venv` con el nombre que prefieras para tu entorno si lo deseas)*.

2.  **Activar el entorno virtual:**
    * **En Windows:**
        ```bash
        .\.venv\Scripts\activate
        ```
    * **En macOS/Linux:**
        ```bash
        source .venv/bin/activate
        ```
    Verás el nombre del entorno (ej. `(.venv)`) al principio de la línea de comandos, indicando que está activo.

3.  **Instalar las dependencias:**
    Con el entorno activado, instala las bibliotecas necesarias:
    ```bash
    pip install numpy matplotlib opencv-contrib-python
    ```

---
## ▶️ Ejecución de los Scripts

Para ejecutar los scripts, primero ingresa a la carpeta de código:
```bash
cd src
```

### Ejercicio 1: Ecualización Local de Histograma (`ej1.py`)

Este script aplica la técnica de ecualización local del histograma a una imagen para resaltar detalles ocultos. El código detecta automáticamente la imagen ubicada en la carpeta `data`.

1. **Ejecutar:**
   ```bash
   python ej1.py
   ```
2.  **Salida**: El script mostrará varias ventanas con la imagen original y los resultados de la ecualización local con diferentes tamaños de ventana. Cierra las ventanas para finalizar. 

### Ejercicio 2: Validación de Formularios (`ej2.py`)

Este script procesa un lote de imágenes de formularios (`formulario_01.png` a `formulario_05.png`) ubicados en la carpeta `data`. Valida automáticamente el estado de los campos y genera un reporte detallado.

1. **Ejecutar:**
   ```bash
   python ej2.py
   ```
   
2.  **Salida**:
    * **Consola**: Imprimirá el resumen de validación para cada formulario. 📝
    * **Archivo CSV**: Creará (o sobrescribirá) un archivo llamado `resultados_formularios.csv` con el estado ('OK' o 'MAL') de cada campo para cada formulario. 
    * **Ventana de Gráficos**: Mostrará una ventana con recortes de las celdas analizadas para el último formulario procesado. Cierra la ventana para finalizar.

---

## Desactivar el Entorno

Cuando termines de trabajar, puedes desactivar el entorno virtual simplemente ejecutando en la terminal:

```bash
deactivate
```

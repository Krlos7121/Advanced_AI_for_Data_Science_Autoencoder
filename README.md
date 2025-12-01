# Autoencoders para la colorización de imagenes de paisajes

## Requisitos

- Python 3.9 en adelante
- TensorFlow
- Pillow
- Matplotlib
- Numpy
- Tkinter
  
## Para correr una query, sigue los siguientes pasos:
- Descarga el modelo de tu elección desde el sitio de Drive: https://drive.google.com/drive/folders/1sH7uGdi-0TlMZR4aMPigxFxLz8NaocJT?usp=sharing
  Nota: (**autoencoder_model.keras** se refiere a la primera implementación del modelo, mientras que **autoencoder_model_sstm.keras** es la implementación mejorada).
- A continuación, descarga el archivo **gui_queries.py** y ejecutalo desde tu terminal.
- Una vez abierta la interfaz gráfica, da clic en la primer opción y selecciona el modelo de tu preferencia.
- Con el segundo botón de la interfaz, selecciona una imagen a blanco y negro (si no tienes ninguna, puedes descargar ejemplos del mismo link de Drive).
- De manera opcional, y si quieres visualizar el score SSIM, selecciona el tercer botón del programa y selecciona la imagen a color equivalente a la que subiste anteriormente.
- Da clic en "Predecir y Mostrar" (:

## Para hacer el ETL y entrenar el modelo desde cero, sigue estos pasos: 
Descarga el dataset de Kaggle: https://www.kaggle.com/datasets/theblackmamba31/landscape-image-colorization/data?select=landscape+Images
- Dentro de la carpeta donde hayas guardado los archivos .ipynb, agrega tus carpetas de imágenes con los nombre **color** y **gray**. 
- Ejecuta el archivo **etl_img.ipynb**
- Crea dos nuevas carpetas en la raíz del directorio en el que estás trabajando, con los nombres **gray_test** y **color_test**, una vez creadas, define el número de imagenes que quieres tomar para tus queries, y seleccionalas por el número más alto al más bajo en el nombre.
  Por ejemplo, si quieres 27 imagenes para probar, tomarías desde la imagen 7128 hasta la 7000, que serán reubicadas a las carpetas de test para **gray** y **color**, respectivamente.
  Con esto, el modelo no será entrenado con esas imagenes.
- Corre **modelo.ipynb** para crear el modelo.
- Corre **gui_queries.py** para ver las predicciones del modelo.

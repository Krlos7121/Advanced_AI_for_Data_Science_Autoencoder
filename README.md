Para correr el código, es importante seguir estos pasos:
Descarga el dataset de Kaggle: https://www.kaggle.com/datasets/theblackmamba31/landscape-image-colorization/data?select=landscape+Images
- Dentro de la carpeta donde hayas guardado los archivos .ipynb, agrega tus carpetas de imágenes con los nombre **color** y **gray**. 
- Ejecuta el archivo **etl_img.ipynb**
- Crea dos nuevas carpetas en la raíz del directorio en el que estás trabajando, con los nombres **gray_test** y **color_test**, una vez creadas, define el número de imagenes que quieres tomar para tus queries, y seleccionalas por el número más alto al más bajo en el nombre.
  Por ejemplo, si quieres 27 imagenes para probar, tomarías desde la imagen 7128 hasta la 7000, que serán reubicadas a las carpetas de test para **gray** y **color**, respectivamente .
- Corre **modelo.ipynb** para crear el modelo.
- Corre **queries.ipynb** para ver las predicciones del modelo.

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Variable global para el modelo
model = None


def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


IMAGE_SIZE = (64, 64)


def load_and_preprocess_image(path, channels=1):
    raw_img = tf.io.read_file(path)
    img_tensor_int = tf.image.decode_jpeg(raw_img, channels=channels)
    rescaled_img = tf.image.resize_with_pad(
        img_tensor_int,
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        antialias=False
    )
    img_tensor_flt = tf.image.convert_image_dtype(
        rescaled_img, tf.float32)  # Normalización [0,1]
    return img_tensor_flt.numpy()


def predict_and_show(gray_img, color_img=None):
    global model
    pred = model.predict(np.expand_dims(gray_img, axis=0))[0]
    fig, axs = plt.subplots(
        1, 3 if color_img is not None else 2, figsize=(13, 4))
    axs[0].imshow(np.squeeze(gray_img), cmap='gray')
    axs[0].set_title('Entrada (Grayscale)')
    axs[0].axis('off')
    axs[1].imshow(pred)
    axs[1].set_title('Predicción')
    axs[1].axis('off')
    ssim_value = None
    if color_img is not None:
        axs[2].imshow(color_img)
        axs[2].set_title('Ground Truth')
        axs[2].axis('off')
        # Calcular SSIM
        ssim_value = tf.image.ssim(
            tf.convert_to_tensor(color_img[np.newaxis, ...], dtype=tf.float32),
            tf.convert_to_tensor(pred[np.newaxis, ...], dtype=tf.float32),
            max_val=1.0
        ).numpy()[0]
        # Incluir SSIM en el plot
        axs[1].text(0.5, -0.12, f"SSIM: {ssim_value:.4f}", fontsize=12,
                    color='blue', ha='center', va='top', transform=axs[1].transAxes)
        print(f"SSIM entre predicción y ground truth: {ssim_value:.4f}")
    plt.tight_layout()
    plt.show()


class App:
    def __init__(self, root):
        self.root = root
        self.root.title('Colorización de Imágenes')
        self.gray_img = None
        self.color_img = None
        self.gray_path = None
        self.color_path = None
        self.model_path = None

        tk.Button(root, text='Seleccionar modelo (.keras)',
                  command=self.select_model).pack(pady=5)
        tk.Button(root, text='Cargar imagen B/N',
                  command=self.load_gray).pack(pady=5)
        tk.Button(root, text='Cargar imagen color (opcional)',
                  command=self.load_color).pack(pady=5)
        tk.Button(root, text='Predecir y mostrar',
                  command=self.predict).pack(pady=10)

    def select_model(self):
        global model
        path = filedialog.askopenfilename(
            filetypes=[('Keras Model', '*.keras')])
        if path:
            try:
                model = tf.keras.models.load_model(
                    path, custom_objects={'ssim_loss': ssim_loss})
                self.model_path = path
                messagebox.showinfo(
                    'Modelo cargado', f'Modelo cargado correctamente:\n{path}')
            except Exception as e:
                messagebox.showerror(
                    'Error', f'No se pudo cargar el modelo:\n{e}')

    def load_gray(self):
        path = filedialog.askopenfilename(
            filetypes=[('Imagen', '*.jpg *.png *.jpeg')])
        if path:
            self.gray_img = load_and_preprocess_image(path, channels=1)
            self.gray_path = path
            messagebox.showinfo(
                'Imagen cargada', 'Imagen B/N cargada correctamente.')

    def load_color(self):
        path = filedialog.askopenfilename(
            filetypes=[('Imagen', '*.jpg *.png *.jpeg')])
        if path:
            self.color_img = load_and_preprocess_image(path, channels=3)
            self.color_path = path
            messagebox.showinfo(
                'Imagen cargada', 'Imagen color cargada correctamente.')

    def predict(self):
        global model
        if model is None:
            messagebox.showerror(
                'Error', 'Primero selecciona un modelo .keras.')
            return
        if self.gray_img is None:
            messagebox.showerror(
                'Error', 'Primero carga una imagen en blanco y negro.')
            return
        predict_and_show(self.gray_img, self.color_img)

        # Limpiar memoria después de generar la predicción
        self.gray_img = None
        self.color_img = None
        self.gray_path = None
        self.color_path = None


if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()

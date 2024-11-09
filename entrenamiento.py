import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import numpy as np
import json
from customtkinter import CTkToplevel, CTkLabel
from utils import get_resource_path, load_keras
import scipy.io
import cv2

train_images, train_labels, test_images, test_labels = None, None, None, None
model = None
test_loss, test_accuracy = None, None

def load_svhn_data(mat_file_path):
    # Cargar el archivo .mat
    svhn_data = scipy.io.loadmat(mat_file_path)
    images = svhn_data['X']  # Contiene las imágenes
    labels = svhn_data['y']  # Contiene las etiquetas

    # Cambiar la etiqueta 10 a 0 (en SVHN, 10 representa el dígito 0)
    labels[labels == 10] = 0

    # Convertir imágenes de (32, 32, 3, N) a (N, 32, 32, 3)
    images = np.moveaxis(images, -1, 0)
    
    # Si es necesario, convertir a escala de grises y normalizar
    images_gray = np.array([cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) / 255.0 for img in images])

    return images_gray, labels

def initialize_svhn_info():
    global train_images, train_labels, test_images, test_labels
    
    # Cargar los datos SVHN
    try:
        train_images, train_labels = load_svhn_data(get_resource_path('Data/train_32x32.mat'))
        test_images, test_labels = load_svhn_data(get_resource_path('Data/test_32x32.mat'))
        extra_images, extra_labels = load_svhn_data(get_resource_path('Data/extra_32x32.mat'))
    
        # Concatenar los conjuntos de entrenamiento y extra
        train_images = np.concatenate([train_images, extra_images], axis=0)
        train_labels = np.concatenate([train_labels, extra_labels], axis=0)
    except Exception as e:
        print(f"Error al cargar los datos SVHN: {e}")
        return False

    # Convertir las etiquetas a formato categórico
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    
    return True

def initialize_mnist_info():
    global train_images, train_labels, test_images, test_labels
    # Cargar los datos MNIST
    try:
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    except Exception as e:
        print(f"Error al cargar los datos MNIST: {e}")
        return False
    
    # Preprocesar los datos: redimensionar y normalizar
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

    # Redimensionar las imágenes a 32x32 para LeNet-5
    train_images = tf.image.resize(train_images, (32, 32)).numpy()
    test_images = tf.image.resize(test_images, (32, 32)).numpy()

    # Convertir las etiquetas a formato categórico
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    
    return True

def initialize_info():
    global train_images, train_labels, test_images, test_labels
    # Cargar los datos MNIST
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Preprocesar los datos: redimensionar y normalizar
    train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

    # Redimensionar las imágenes a 32x32 para LeNet-5
    train_images = tf.image.resize(train_images, (32, 32)).numpy()
    test_images = tf.image.resize(test_images, (32, 32)).numpy()

    # Convertir las etiquetas a formato categórico
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

def show_train_images():
    global train_images, train_labels, test_loss, test_accuracy
    # Buscar 5 imágenes de cada dígito (0-9) en el conjunto de entrenamiento
    num_classes = 10
    num_images_per_class = 5
    fig, axes = plt.subplots(num_classes, num_images_per_class, figsize=(8, 8))

    # Convertir las imágenes a 2D para visualizarlas
    train_images_2d = train_images.reshape(-1, 32, 32)
    # Recorrer las clases (dígitos del 0 al 9)
    for i in range(num_classes):
        # Encontrar las primeras 5 imágenes de cada dígito
        indices = np.where(np.argmax(train_labels, axis=1) == i)[0][:num_images_per_class]
        for j, index in enumerate(indices):
            axes[i, j].imshow(train_images_2d[index], cmap='gray')
            axes[i, j].axis('off')

    # reduce the vertical space in the plot at the beginning and end
    plt.subplots_adjust(hspace=0.1, wspace=0.1)

    return fig

def train_leNet():
    global model, train_images, train_labels, test_images, test_labels
    # Definir la arquitectura de LeNet-5
    model = models.Sequential()
    model.add(keras.Input(shape=(32, 32, 1)))
    model.add(layers.Conv2D(6, (5, 5), activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(16, (5, 5), activation='relu'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation='relu'))
    model.add(layers.Dense(84, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    # Compilar el modelo
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(
        monitor='val_accuracy',   # Métrica a monitorear (val_accuracy = precisión en los datos de validación)
        patience=3,               # Número de épocas sin mejora para detener el entrenamiento
        min_delta=0.001,          # Umbral mínimo de mejora entre épocas
        mode='max',               # 'max' porque buscamos maximizar la precisión
        verbose=1                 # Imprime mensajes cuando se activa el early stopping
    )

    # Entrenar el modelo
    model.fit(train_images, train_labels, epochs=1000, batch_size=128, validation_data=(test_images, test_labels), callbacks=[early_stopping])

    # Evaluar el modelo
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    
    model_path = get_resource_path('Data/lenet_5_model.keras')
    model.save(model_path)
    
    return model

def train_leNet_old():
    global model, train_images, train_labels, test_images, test_labels
    # Definir la arquitectura de LeNet-5
    model = models.Sequential()
    model.add(keras.Input(shape=(32, 32, 1)))
    model.add(layers.Conv2D(6, (5, 5), activation='tanh'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(16, (5, 5), activation='tanh'))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation='tanh'))
    model.add(layers.Dense(84, activation='tanh'))
    model.add(layers.Dense(10, activation='softmax'))
    # Compilar el modelo
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(
        monitor='val_accuracy',   # Métrica a monitorear (val_accuracy = precisión en los datos de validación)
        patience=3,               # Número de épocas sin mejora para detener el entrenamiento
        min_delta=0.001,          # Umbral mínimo de mejora entre épocas
        mode='max',               # 'max' porque buscamos maximizar la precisión
        verbose=1                 # Imprime mensajes cuando se activa el early stopping
    )

    # Entrenar el modelo
    model.fit(train_images, train_labels, epochs=1000, batch_size=128, validation_data=(test_images, test_labels), callbacks=[early_stopping])

    # Evaluar el modelo
    test_loss, test_accuracy = model.evaluate(test_images, test_labels)
    
    model_path = get_resource_path('Data/lenet_5_model.keras')
    model.save(model_path)
    
    return model

def kernel_values(model):
    # Recorrer las capas del modelo
    info_capas = ""
    for layer in model.layers:
        if isinstance(layer, Conv2D):
            # Obtener los pesos de la capa convolucional (filtros y sesgos)
            filters, biases = layer.get_weights()

            num_filters = filters.shape[3]  # Número de filtros
            filter_size = filters.shape[0]  # Tamaño de los filtros (supongamos que es cuadrado)
            input_channels = filters.shape[2]  # Número de canales de entrada (1 si es escala de grises, 3 si es RGB)

            info_capas += f"\nCapa {layer.name}:\n"
            info_capas += f"Número de filtros: {num_filters}\n" 
            info_capas += f"Tamaño de los filtros: {filter_size}x{filter_size}\n"
            info_capas += f"Número de canales de entrada: {input_channels}\n"
            info_capas += "Valores de los filtros (kernels):\n"
            # Imprimir los valores de cada filtro
            for i in range(num_filters):
                info_capas += f"Filtro {i + 1}:\n"
                for j in range(input_channels):  # Si tiene más de un canal (ej. RGB), recorrer cada canal
                    info_capas += f" Canal {j + 1}:\n {json.dumps(filters[:, :, j, i].tolist(), indent=4)}\n"

            info_capas += "Valores de los sesgos (biases):\n"
            info_capas += f"{json.dumps(biases.tolist(), indent=4)}\n"
            
    return info_capas

def get_summary(model):
    
    if not model.built:
        model.build((None, 32, 32, 1))
    
    summary = { "layers": [] }
    print(model.summary())
    
    # Save the summary info in a dictionary
    for layer in model.layers:
        layer_i ={
            "name": layer.name,
            "type": layer.__class__.__name__,
            "parameters": layer.count_params()
        }
     
        summary["layers"].append(layer_i)
        
    summary["layers"][0]["output_shape"] = "(None, 28, 28, 1)"
    summary["layers"][1]["output_shape"] = "(None, 14, 14, 6)"
    summary["layers"][2]["output_shape"] = "(None, 10, 10, 6)"
    summary["layers"][3]["output_shape"] = "(None, 5, 5, 16)"
    summary["layers"][4]["output_shape"] = "(None, 1, 1, 120)"
    summary["layers"][5]["output_shape"] = "(None, 120)"
    summary["layers"][6]["output_shape"] = "(None, 84)"
    summary["layers"][7]["output_shape"] = "(None, 10)"
    
    return summary

def set_model(mod):
    global model
    model = mod
    
def set_default_model():
    global model
    try:
        model_path = get_resource_path('Data/lenet_5_model.keras')
        model = load_model(model_path)
        print("Modelo cargado correctamente")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        model = None

def load_p_model():
    global model
    model_temp, filename = load_keras(extension = '.keras')
    if model_temp is not None:
        model = model_temp
        # create a top level to show the info of model charged
        top = CTkToplevel()
        top.title("Model Info")
        top.geometry("500x50")
        top.resizable(False, False)
        top.attributes("-topmost", True)
        # create a label to show the info of the model
        label = CTkLabel(top, text = "Modelo Cargado Correctamente", font = ("courier new", 20, "bold"), text_color="green")
        label.pack()
        
        top.after(3000, top.destroy)   
        
def predict(image):
    global model
    
    # Predict the digit
    image = image.reshape(1, 32, 32, 1)
    prediction = model.predict(image)
    digit = np.argmax(prediction)
    
    # Verify if more than one digit is detected above 1 x 10^-10, then return "Desconocido"
    digits = np.where(prediction > 1e-10)[0]
    if len(digits) > 1:
        return prediction, "Desconocido"

    return prediction, digit

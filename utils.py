# file for store functions that are used in differents files of the proyect

import os # Import the os module to interact with the operating system
import sys # Import the sys module to interact with the Python interpreter
import random # Import the random module to generate random numbers
import math # Import the math module to perform mathematical operations
import json # Import the json module to work with JSON files
import shutil # Import the shutil module to perform file operations
import webbrowser # Import the webbrowser module to open web pages
from tkinter import filedialog # Import the filedialog module to open file dialogs
from keras.models import load_model # Import the load_model function from the keras.models module
import cv2 # Import the cv2 module to work with images


def get_resource_path(relative_path):
    """
    Get absolute path to resource, works for dev and for PyInstaller
    params:
        relative_path: relative path to the resource
    return: absolute path to the resource
    """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def open_link(event = None, link = None):
    """
    Open the Link in the default browser.
    
    Parameters:
    event (tk.Event): Event object.
    """
    webbrowser.open(link)    

def download_json(filename, extension = ".keras", data = None, directory = "Data"):
    """
    Save a file in a new location.
    
    Parameters:
    filename (str): Name of the file.
    extension (str): Extension of the file.
    """
    file_path = filedialog.asksaveasfilename(defaultextension=extension, filetypes=[("Keras files", f"*{extension}")], initialfile=f"{filename}{extension}")
    if data:    
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
            print(f"Archivo guardado en: {file_path}")
    elif file_path:
        json_path = get_resource_path(f"{directory}/{filename}{extension}")
        shutil.copy(json_path, file_path)  # Copiar el archivo a la nueva ubicación
        print(f"Archivo guardado en: {file_path}")
    
def load_keras(filename = None, extension = ".keras", directory = "Data"):
    """
    Load a Keras file.
    
    Parameters:
    filename (str): Name of the file.
    extension (str): Extension of the file.
    
    Returns:
    data_keras  Keras data.
    """
    data_keras = None
    if filename is None:
        file = filedialog.askopenfilename(filetypes=[("Keras files", f"*{extension}")])
        if file:
            file_name = get_resource_path(file)
            data_keras = load_model(file_name)
            return data_keras, file_name
        else:
            return None, None

    return None, None        

def load_images(is_folder):
    """
    Load images from a folder or a single image.
    
    Parameters:
    is_folder (bool): True if the user wants to load a folder, False if the user wants to load a single image.
    
    Returns:
    images (list): List with the images loaded.
    """
    images = []
    try:
        if is_folder:
            folder = filedialog.askdirectory()
            folder = get_resource_path(folder)
            if folder:
                for filename in os.listdir(folder):
                    img_path = os.path.join(folder, filename)
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
                    if img is not None:
                        images.append(img)
        else:
            img_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.ppm;*.pgm;*.pbm;*.webp")]
)
            if img_path:
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
                if img is not None:
                    images.append(img)
        return images
    except Exception as e:
        print("Error loading images: ", e)
        return images

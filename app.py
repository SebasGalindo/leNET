import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
from utils import download_json, get_resource_path, open_link, load_images
import entrenamiento as e
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

ctk.set_appearance_mode("dark")
title_font, sub_title_font, text_font, code_font = None, None, None, None
main_window, ctk_top_level = None, None
content_frame, top_frame = None, None
train_thread, kernel_values, kernel_text = None, "", None
download_button = None
prepared_image = None
transmision_activa = False
cap, video_label = None, None
db_name = ""

def grid_setup(frame):
    for i in range(12): 
        frame.grid_rowconfigure(i, weight=1)
        frame.grid_columnconfigure(i, weight=1)
    return frame

def main_window_creation():
    global main_window, title_font, sub_title_font, text_font, content_frame, code_font
    
    main_window = ctk.CTk(fg_color="#00482b")
    main_window.title("LeNet-5")
    
    screen_width = main_window.winfo_screenwidth()
    screen_height = main_window.winfo_screenheight()
    
    window_width = 1200
    window_height = 700
    
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2

    # Establece la posición y el tamaño de la ventana
    main_window.geometry(f"{window_width}x{window_height}+{x}+{y}")
    main_window.resizable(False, False)
    icon_path = get_resource_path("Resources/brand_logo.ico")
    main_window.iconbitmap(icon_path)
    
    for i in range(12):
        if i >= 2:
            main_window.grid_rowconfigure(i, weight=3)
        else:
            main_window.grid_rowconfigure(i, weight=1)
        main_window.grid_columnconfigure(i, weight=1)


    title_font = ctk.CTkFont(family="courier new", size=24, weight="bold")
    sub_title_font = ctk.CTkFont(family="courier new", size=16, weight="bold")
    text_font = ctk.CTkFont(family="comfortaa", size=14, weight="bold")
    code_font = ctk.CTkFont(family="consolas", size=20, weight="bold")

    add_menu(main_window)
    
    add_sub_menu(main_window)
    
    # Content frame
    content_frame = ctk.CTkScrollableFrame(master=main_window, corner_radius=0, fg_color="#00482b", border_width=0)
    content_frame.grid(row=2, column=0, columnspan=12, sticky="nsew", padx=0, pady=0, rowspan=10)
    content_frame = grid_setup(content_frame)
    
    main_window.mainloop()

def add_menu(window):
    global title_font, sub_title_font, text_font
    # Vertical frame in row 1 for the menu
    menu_frame = ctk.CTkFrame(master=window, fg_color="#00482b", corner_radius=0, border_width=0)
    menu_frame.grid(row=0, column=0, columnspan=12, sticky="nsew", padx=0, pady=0)
    
    menu_frame = grid_setup(menu_frame)
    
    # Add Udec Logo
    logo_path = get_resource_path("Resources/logo_UdeC_Blanco.png")
    udec_img = Image.open(logo_path)
    width, height = udec_img.size
    width, height = int(width/12), int(height/12)
    logo_udec = ctk.CTkImage(light_image=udec_img, dark_image=udec_img, size=(width, height))
    
    logo_label = ctk.CTkLabel(master=menu_frame, image=logo_udec, bg_color="#00482b", font=title_font, compound="center", text= "")
    logo_label.grid(row=0, column=0, columnspan=2, sticky="nsw", padx = 20, pady = 10)
    
    title_label = ctk.CTkLabel(master=menu_frame, text="LeNet-5", bg_color="#00482b", font=title_font, text_color="#ffffff")
    title_label.grid(row=0, column=2, columnspan=8, sticky="nsew", padx = 10, pady = 10)
    authors_names_label = ctk.CTkLabel(master=menu_frame, text="John Sebastián Galindo Hernández\nMiguel Ángel Moreno Beltrán", bg_color="#00482b", font=sub_title_font, text_color="#ffffff", justify="right")
    authors_names_label.grid(row=0, column=10, columnspan=2, sticky="nse", padx = 10, pady = 10)

def add_sub_menu(window):
    global title_font, sub_title_font, text_font
    
    # Vertical frame in row 2 for the sub-menu
    sub_menu_frame = ctk.CTkFrame(master=window, fg_color="#00482b", corner_radius=0, border_width=0)
    sub_menu_frame.grid(row=1, column=0, columnspan=12, sticky="nsew", padx=0, pady=0, rowspan=1)
    
    for i in range(8):
        sub_menu_frame.grid_columnconfigure(i, weight=1)

    sub_menu_frame.grid_rowconfigure(0, weight=1)
    
    # Training and testing data button
    data_button = ctk.CTkButton(master=sub_menu_frame, text="Datos de entrenamiento y prueba",height=30, fg_color="#daaa00", hover_color="#e5c44d", text_color="black", font=sub_title_font, command=show_data_info)
    data_button.grid(row=0, column=0, columnspan=2, sticky="nsew", padx = 10, pady = 12)
    
    # Train button
    train_button = ctk.CTkButton(master=sub_menu_frame, text="Entrenar modelo", fg_color="#daaa00",height=30, text_color="black", hover_color="#e5c44d", font=sub_title_font, command=train_model)
    train_button.grid(row=0, column=2, columnspan=1, sticky="nsew", padx = 10, pady = 12)
    
    # Test button
    test_button = ctk.CTkButton(master=sub_menu_frame, text="Probar modelo", fg_color="#daaa00", height=30, text_color="black", hover_color="#e5c44d", font=sub_title_font, command=test_model)
    test_button.grid(row=0, column=3, columnspan=1, sticky="nsew", padx = 10, pady = 12)
    # Capture button
    capture_button = ctk.CTkButton(master=sub_menu_frame, text="Capturar video", fg_color="#daaa00", height=30, text_color="black", hover_color="#e5c44d", font=sub_title_font, command=capture_video)
    capture_button.grid(row=0, column=4, columnspan=1, sticky="nsew", padx = 10, pady = 12)
    
    # documentation button
    document_path = get_resource_path("Resources/pdf_logo.png")
    document_img = Image.open(document_path)
    width, height = document_img.size
    width, height = int(width/11), int(height/11)
    document_img = ctk.CTkImage(light_image=document_img, dark_image=document_img, size=(width, height))
    document_label = ctk.CTkLabel(master=sub_menu_frame, image=document_img, bg_color="#00482b", font=title_font, compound="center", text= "")
    document_label.grid(row=0, column=5, columnspan=1, sticky="nsew", padx = 10, pady = 2)
    document_label.bind("<Button-1>", lambda e: open_link(link="youtube.com"))
    document_label.configure(cursor="hand2")
    
    # presentation buttonw
    presentation_path = get_resource_path("Resources/ppt_logo.png")
    presentation_img = Image.open(presentation_path)
    width, height = presentation_img.size
    width, height = int(width/11), int(height/11)
    presentation_img = ctk.CTkImage(light_image=presentation_img, dark_image=presentation_img, size=(width, height))
    presentation_label = ctk.CTkLabel(master=sub_menu_frame, image=presentation_img, font=title_font, compound="center", text= "")
    presentation_label.grid(row=0, column=6, columnspan=1, sticky="nsew", padx = 10, pady = 2)
    presentation_label.bind("<Button-1>", lambda e: open_link(link="google.com"))
    presentation_label.configure(cursor="hand2")
    
    # github button
    github_path = get_resource_path("Resources/github_logo.png")
    github_img = Image.open(github_path)
    width, height = github_img.size
    width, height = int(width/11), int(height/11)
    github_img = ctk.CTkImage(light_image=github_img, dark_image=github_img, size=(width, height))
    github_label = ctk.CTkLabel(master=sub_menu_frame, image=github_img, bg_color="#00482b", font=title_font, compound="center", text= "")
    github_label.grid(row=0, column=7, columnspan=1, sticky="nsew", padx = 10, pady = 2)
    github_label.bind("<Button-1>", lambda e: open_link(link="github.com"))
    github_label.configure(cursor="hand2")

def train_model():
    global content_frame, train_thread, kernel_values, kernel_text, download_button, transmision_activa, cap
    
    transmision_activa = False
    if cap is not None:
        cap.release()
    
    if e.train_images is None:
        initialize_info_frame()
    
    if e.train_images is None:
        return
    
    # Train the model in a new thread    
    train_thread = threading.Thread(target=e.train_leNet)
    train_thread.start()
    
    model = e.model
    check_thread()

    # forget the previous content in content_frame
    for widget in content_frame.winfo_children():
        widget.destroy()
        
    # Create a frame to show the kernel values with a scrollbar and a CTKTextbox to show the values
    kernel_frame = ctk.CTkFrame(master=content_frame, fg_color="#ffffff", height=500 , corner_radius=8, border_width=0)
    kernel_frame.grid(row=0, column=0, columnspan=4, sticky="nsew", padx=10, pady=10, rowspan=12)
    kernel_frame = grid_setup(kernel_frame)
    
    # Create a font of 24 px for the start training info in slateblue color
    start_training_font = ctk.CTkFont(family="comfortaa", size=24, weight="bold")
    
    kernel_text = ctk.CTkTextbox(master=kernel_frame, fg_color="#ffffff", font=code_font, text_color="#000000", border_width=0, corner_radius=8, wrap="word", height=500)
    kernel_text.grid(row=0, column=0, columnspan=12, sticky="nsew", padx=10, pady=10, rowspan=12)

    kernel_text.tag_config("title", cnf = {"font": start_training_font}, foreground="slateblue")
    kernel_text.delete(1.0, "end")
    kernel_text.insert("end", "Entrenando modelo... \nPara ver la información en tiempo real del entrenamiento mire su consola", "title")
    
    # Button for download the model
    download_button = ctk.CTkButton(master=content_frame, text="Descargar modelo", fg_color="#daaa00", height=30, text_color="black", hover_color="#e5c44d", font=sub_title_font)
    download_button.configure(command=lambda: download_json(filename="lenet_5_model", extension=".keras"), state = "disabled")
    
def check_thread():
    global train_thread, kernel_values, main_window, kernel_text, code_font
    if not train_thread.is_alive():  # Si el hilo ha terminado
        kernel_values = e.kernel_values(e.model)
        kernel_text.delete(1.0, "end")
        kernel_text.tag_config("code", cnf = {"font": code_font}, foreground="#002416")
        kernel_text.insert("end", kernel_values, "code")
        put_model_summary()
    else:
        main_window.after(100, check_thread) 

def put_model_summary():
    global content_frame, code_font, download_button
    summary = e.get_summary(e.model)

    # Summary frame
    summary_frame = ctk.CTkTextbox(master=content_frame, fg_color="#ffffff", corner_radius=8, border_width=0)
    summary_frame.grid(row=0, column=4, columnspan=4, sticky="nsew", padx=10, pady=10, rowspan=12)
    
    start_training_font = ctk.CTkFont(family="comfortaa", size=20, weight="bold")
    start_training_font = ctk.CTkFont(family="comfortaa", size=18, weight="bold")
    summary_frame.tag_config("title", cnf = {"font": start_training_font}, foreground="#1a5a40")
    summary_frame.tag_config("header", cnf = {"font": start_training_font}, foreground="#1a5a40")
    summary_frame.tag_config("code", cnf = {"font": code_font}, foreground="#002416")

    separator = "-" * 80
    summary_frame.delete(1.0, "end")
    summary_frame.insert("end", "Resumen del modelo\n", "title")
    summary_frame.insert("end", f"{separator}\n", "header")
    for line in summary["layers"]:
        # name of the layer
        summary_frame.insert("end", "Nombre: ", "header")
        summary_frame.insert("end", f"{line["name"]}\n", "code")
        
        # type of the layer
        summary_frame.insert("end", "Tipo: ", "header")
        summary_frame.insert("end", f"{line["type"]}\n", "code")

        # output shape of the layer
        summary_frame.insert("end", "Salida: ", "header")
        summary_frame.insert("end", f"{line["output_shape"]}\n", "code")
        
        # parameters of the layer
        summary_frame.insert("end", "Parámetros: ", "header")
        summary_frame.insert("end", f"{line["parameters"]}\n", "code")
        
        summary_frame.insert("end", f"{separator}\n", "header")

    download_button.grid(row=0, column=11, columnspan=1, sticky="new", padx = 10, pady = 12)
    download_button.configure(state = "normal")
        
def test_model():
    global content_frame, code_font, sub_title_font, transmision_activa, cap
    
    transmision_activa = False
    if cap is not None:
        cap.release()
    
    # forget the previous content in content_frame
    for widget in content_frame.winfo_children():
        widget.destroy()
        
    if e.model is None:
        e.set_default_model()
        e.model.summary()
    
    # Button for load a personalizated model
    load_button = ctk.CTkButton(master=content_frame, text="Cargar modelo", fg_color="#daaa00", height=30, text_color="black", hover_color="#e5c44d", font=sub_title_font, command=e.load_p_model)                  
    load_button.grid(row=0, column=0, columnspan=2, sticky="nsew", padx = 10, pady = 12)
    
    # Button for load a image
    load_img_button = ctk.CTkButton(master=content_frame, text="Cargar imagen", fg_color="#daaa00", height=30, text_color="black", hover_color="#e5c44d", font=sub_title_font, command=load_image)
    load_img_button.grid(row=0, column=2, columnspan=2, sticky="nsew", padx = 10, pady = 12)
    
def load_image():
    global content_frame, prepared_image
    
    images = load_images(is_folder=False)
    width = images[0].shape[1]
    height = images[0].shape[0]
    max_height = 400
    ratio = max_height / height
    width = int(width * ratio)
    height = int(height * ratio)
    normal_image = cv2.resize(images[0], (width, height), interpolation = cv2.INTER_AREA)
    
    # Put the image in the content_frame in a label
    image = Image.fromarray(normal_image)
    image = ctk.CTkImage(light_image=image, dark_image=image, size=(width, height))
    image_label = ctk.CTkLabel(master=content_frame, image=image, bg_color="#00482b", text="")
    image_label.grid(row=1, column=0, columnspan=4, sticky="new", padx=10, pady=10, rowspan=11)    
    
    # Prepare the image
    prepared_image = prepare_image(normal_image)

    # Put the prepared image in the content_frame in a label
    prepared_image_temp = Image.fromarray(prepared_image)
    prepared_image_temp = ctk.CTkImage(light_image=prepared_image_temp, dark_image=prepared_image_temp, size=(32, 32))
    prepared_image_label = ctk.CTkLabel(master=content_frame, image=prepared_image_temp, bg_color="#00482b", text="")
    prepared_image_label.grid(row=1, column=4, columnspan=2, sticky="new", padx=10, pady=10, rowspan=11)
    
    # Button for predict the image
    predict_button = ctk.CTkButton(master=content_frame, text="Predecir imagen", fg_color="#daaa00", height=30, text_color="black", hover_color="#e5c44d", font=sub_title_font)
    predict_button.grid(row=1, column=6, columnspan=2, sticky="nw", padx = 10, pady = 12)
    predict_button.configure(command=lambda: predict_image(prepared_image))

def prepare_image(image):
    """
        Function to prepare the image to be predicted by the model
        this function resize the image to 32x32 and convert it to grayscale
        if the database is mnist, the colors are inverted
        
    """
    
    global db_name
    
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # invert the colors
    if db_name == "mnist":
        gray = cv2.bitwise_not(gray)

    gray_resized = cv2.resize(gray, (32, 32))
    
    return gray_resized
    
def predict_image(prepared_image):
    global content_frame, code_font

    # Do Prediction
    prediction, digit = e.predict(prepared_image)

    # Put the information in a textbox
    results_textbox = ctk.CTkTextbox(master=content_frame, fg_color="#ffffff", corner_radius=8, border_width=0)
    results_textbox.grid(row=2, column=6, columnspan=6, sticky="nsew", padx=10, pady=10, rowspan=10)
    
    start_training_font = ctk.CTkFont(family="comfortaa", size=20, weight="bold")
    start_training_font = ctk.CTkFont(family="comfortaa", size=18, weight="bold")
    results_textbox.tag_config("title", cnf = {"font": start_training_font}, foreground="#1a5a40")
    results_textbox.tag_config("header", cnf = {"font": start_training_font}, foreground="#1a5a40")
    results_textbox.tag_config("code", cnf = {"font": code_font}, foreground="#002416")

    separator = "-" * 73
    results_textbox.delete(1.0, "end")
    results_textbox.insert("end", "Predicción del modelo\n", "title")
    results_textbox.insert("end", f"{separator}\n", "header")
    results_textbox.insert("end", "Predicción: ", "header")
    results_textbox.insert("end", f"{digit}\n", "code")
    results_textbox.insert("end", "Probabilidades: ", "header")
    txt_prob = ""
    for i, prob in enumerate(prediction[0]):
        txt_prob += f"{i}: {prob * 100:.6f} %\n"
        
    results_textbox.insert("end", f"{txt_prob}\n", "code")
    results_textbox.insert("end", f"{separator}\n", "header")
    
def capture_video():
    global content_frame, transmision_activa, cap, video_label
    
    transmision_activa = True
    
    # Delete the previous content in the content_frame
    for widget in content_frame.winfo_children():
        widget.destroy()   
        
    # URL de la cámara IP con usuario y contraseña
    url = url_peticion()

    if url is None or url == "":
        show_error_info("Error al obtener la URL de la cámara IP")
        return 

    if url == "0":
        url = int(url)
        
    # Conectar a la cámara IP
    try: 
        cap = cv2.VideoCapture(url )
    except Exception as e:
        show_error_info("Error al conectar con la cámara IP")
        print(f"Error al conectar con la cámara IP: {e}")
        return
    
    # label for the video
    video_label = ctk.CTkLabel(master=content_frame, text="", fg_color="#ffffff", width=640, height=480)
    video_label.grid(row=0, column=0, columnspan=12, sticky="snew", padx=10, pady=10, rowspan=10)
    
    mostrar_video()
    
def mostrar_video():
    global transmision_activa, video_label, cap
    
    if transmision_activa:
        ret, frame = cap.read()
        
        # Redimensionar el frame a un tamaño más pequeño (por ejemplo, 640x480)
        frame = cv2.resize(frame, (840, 680))
        
        # Dibujar un recuadro donde mostrar el dígito (por ejemplo, una región en el centro de la imagen)
        cv2.rectangle(frame, (100, 100), (500, 500), (29, 41, 216), 2)
        
        # Extraer la región de interés (ROI) donde se espera que esté el dígito
        roi = frame[100:400, 100:400]
        
        roi = prepare_image(roi)
        p, digit = e.predict(roi)
        
        # Mostrar la predicción en la imagen
        cv2.putText(frame, f'Digito: {digit}', (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (216, 29, 69), 2)
        
        if ret:
            # Convertir el frame a RGB y luego a formato de Tkinter
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            
            video_label.configure(image=imgtk)
        
        # Continuar actualizando el frame cada 10 ms
        video_label.after(10, mostrar_video)
    else:
        # Cuando la transmisión está desactivada, limpiar el Label
        video_label.destroy()    
      
def on_key_press(event):
    global transmision_activa
    if event.char == 'q':
        transmision_activa = False

def show_data_info():   
    global content_frame, transmision_activa
    
    transmision_activa = False
    if cap is not None:
        cap.release()
    
    if e.train_images is None:
        initialize_info_frame()
    
    if e.train_images is None:
        return
    
    # Delete the previous content in the content_frame
    for widget in content_frame.winfo_children():
        widget.destroy()
    
    figure = e.show_train_images()
    
    # Create the canvas to show the figure
    canvas = FigureCanvasTkAgg(figure, content_frame)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0, columnspan=12, sticky="nsew", padx=10, pady=10)

def initialize_info_frame():
    """
        CTK Top Level to Select one of the two databases (MNIST or SVHN)
    """
    global ctk_top_level, top_frame
    
    if ctk_top_level is not None:
        ctk_top_level.destroy()
    
    ctk_top_level = ctk.CTkToplevel()
    ctk_top_level.title("Elección de Base de datos")
    
    screen_width = ctk_top_level.winfo_screenwidth()
    screen_height = ctk_top_level.winfo_screenheight()
    
    window_width = 300
    window_height = 200
    
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2

    # Establece la posición y el tamaño de la ventana
    ctk_top_level.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    ctk_top_level.resizable(False, False)
    ctk_top_level.iconbitmap(get_resource_path("Resources/brand_logo.ico"))
    ctk_top_level.attributes("-topmost", True)
    
    ctk_top_level.grid_rowconfigure(0, weight=1)
    ctk_top_level.grid_columnconfigure(0, weight=1)
    
    # Frame for the buttons
    top_frame = ctk.CTkFrame(master=ctk_top_level, fg_color="#00482b", corner_radius=0, border_width=0)
    top_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
    top_frame = grid_setup(top_frame)
    
    # Button for MNIST
    mnist_button = ctk.CTkButton(master=top_frame, text="MNIST", fg_color="#daaa00", height=30, text_color="black", hover_color="#e5c44d", font=sub_title_font)
    mnist_button.grid(row=0, column=0, columnspan=6, sticky="nsw", padx = 10, pady = 12)
    mnist_button.configure(command= lambda: initialize_info("mnist"))

    # Button for SVHN
    svhn_button = ctk.CTkButton(master=top_frame, text="SVHN", fg_color="#daaa00", height=30, text_color="black", hover_color="#e5c44d", font=sub_title_font)
    svhn_button.grid(row=0, column=6, columnspan=6, sticky="nse", padx = 10, pady = 12)
    svhn_button.configure(command= lambda: initialize_info("svhn"))
    
    ctk_top_level.wait_window(ctk_top_level)
    
def initialize_info(database_name):
    global ctk_top_level, db_name, top_frame
    is_info_charged = False
    
    if database_name == "mnist":
        is_info_charged = e.initialize_mnist_info()
    elif database_name == "svhn":
        is_info_charged = e.initialize_svhn_info()
    
    if is_info_charged:
        db_name = database_name
        #Label to show the charged successfully status
        label = ctk.CTkLabel(top_frame, text="Base de datos cargada correctamente", font=sub_title_font, text_color="seagreen", wraplength=200)
        label.grid(row=1, column=0, columnspan=12, sticky="nsew", padx = 10, pady = 12)
        ctk_top_level.after(1000, ctk_top_level.destroy)  
    else:
        #Label to show the error status
        label = ctk.CTkLabel(top_frame, text="Error al cargar la base de datos", font=sub_title_font, text_color="darkred", wraplength=200)
        label.grid(row=1, column=0, columnspan=12, sticky="nsew", padx = 10, pady = 12)
    
def url_peticion():
    """
        Function to get the URL of the IP camera,
        the default value is 0,
        if is 0, the camera of the computer is used
        This function creates a CTK Top Level to ask the URL using a CTK Entry
    """

    global ctk_top_level, top_frame
    
    if ctk_top_level is not None:
        ctk_top_level.destroy()
    
    ctk_top_level = ctk.CTkToplevel()
    ctk_top_level.title("URL de la cámara IP")
    
    screen_width = ctk_top_level.winfo_screenwidth()
    screen_height = ctk_top_level.winfo_screenheight()
    
    window_width = 400
    window_height = 200
    
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2

    # Establece la posición y el tamaño de la ventana
    ctk_top_level.geometry(f"{window_width}x{window_height}+{x}+{y}")
    
    ctk_top_level.resizable(False, False)
    ctk_top_level.iconbitmap(get_resource_path("Resources/brand_logo.ico"))
    ctk_top_level.attributes("-topmost", True)
    
    ctk_top_level.grid_rowconfigure(0, weight=1)
    ctk_top_level.grid_columnconfigure(0, weight=1)
    
    # Frame for the content
    top_frame = ctk.CTkFrame(master=ctk_top_level, fg_color="#00482b", corner_radius=0, border_width=0)
    top_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
    top_frame = grid_setup(top_frame)
    
    # Label for the URL
    url_label = ctk.CTkLabel(master=top_frame, text="URL de la cámara IP \n (0 es la camara del computador)", text_color="#ffffff", font=sub_title_font)
    url_label.grid(row=0, column=0, columnspan=12, sticky="nsew", padx=10, pady=12)
    
    # Entry for the URL
    url_entry = ctk.CTkEntry(master=top_frame, fg_color="#ffffff", text_color="black", corner_radius=4, border_width=0)
    url_entry.grid(row=1, column=0, columnspan=12, sticky="nsew", padx=10, pady=12)
    url_entry.insert(0, "0")
    
    # Function to get the URL value
    def aceptar_url():
        nonlocal url_val
        url_val = url_entry.get() 
        ctk_top_level.destroy()
    
    # Button to accept the URL
    accept_button = ctk.CTkButton(master=top_frame, text="Aceptar", fg_color="#daaa00", height=30, text_color="black", hover_color="#e5c44d", font=sub_title_font)
    accept_button.grid(row=2, column=4, columnspan=4, sticky="nsew", padx=10, pady=12)
    accept_button.configure(command=aceptar_url)
    
    url_val = None 
    ctk_top_level.wait_window(ctk_top_level) 
    return url_val 
 
def show_error_info(error_message):
    """
        Function to show an error message in a CTK Top Level
    """
    global ctk_top_level, top_frame, sub_title_font
    
    if ctk_top_level is not None:
        ctk_top_level.destroy()
    
    ctk_top_level = ctk.CTkToplevel()
    ctk_top_level.title("Error")

    screen_width = ctk_top_level.winfo_screenwidth()
    screen_height = ctk_top_level.winfo_screenheight()
    
    window_width = 400
    window_height = 150
    
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2

    # Establece la posición y el tamaño de la ventana
    ctk_top_level.geometry(f"{window_width}x{window_height}+{x}+{y}")

    ctk_top_level.resizable(False, False)
    ctk_top_level.iconbitmap(get_resource_path("Resources/brand_logo.ico"))
    ctk_top_level.attributes("-topmost", True)
    
    ctk_top_level.grid_rowconfigure(0, weight=1)
    ctk_top_level.grid_columnconfigure(0, weight=1)
    
    # Frame for the content
    top_frame = ctk.CTkFrame(master=ctk_top_level, fg_color="#00482b", corner_radius=0, border_width=0)
    top_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=0)
    top_frame = grid_setup(top_frame)
    
    # Label for the error message
    error_label = ctk.CTkLabel(master=top_frame, text=error_message, text_color="red"  ,font=sub_title_font, wraplength=300, justify="center")
    error_label.grid(row=0, column=0, columnspan=12, sticky="nsew", padx = 10, pady = 12, rowspan=12)
    
    ctk_top_level.after(2000, ctk_top_level.destroy)
    
if __name__ == '__main__':
    if e.model is None:
        e.set_default_model()
    main_window_creation()
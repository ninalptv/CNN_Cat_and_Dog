import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import tensorflow as tf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow import keras
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter.filedialog import askopenfilename
import matplotlib.image as mpimg
from keras.utils import load_img
from keras.utils import img_to_array
import numpy as np

model = keras.models.load_model('cnn_cat_and_dog')
# Обработчик (событие) нажатия на кнопку "Выбрать"
def open_image():
    # Запускаем диалог выбора файла
    global filepath
    filepath = askopenfilename(
        filetypes=[("JPG-file", "*.jpg")]
    )
    if not filepath:
        return

    window.chart.clear()
    ax1 = window.chart.add_subplot(111, xticks=[], yticks=[])
    img = mpimg.imread(filepath)
    ax1.imshow(img)
    window.chart_canvas.draw()
    btn_start.config(state="normal")


def start_work():
    test_image = load_img(filepath, target_size=(200, 200))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image)
    if (result >= 0.4):
        tk.Label(window, font=("Arial", 14),text=f" Это Собачка").place(x=10, y=200)
    else:
        tk.Label(window, font=("Arial", 14),text=f" Это Кошечка").place(x=10, y=200)


# Создаем главное окно
window = tk.Tk()
window.title("Кошечки Собачки")
window.geometry("600x450")
window.resizable(True, True)
window.minsize(400, 250)

# Настраиваем его разметку
window.rowconfigure(0,minsize=400,weight=1)
window.columnconfigure(1, minsize=250, weight=1)
window.chart=plt.Figure(figsize=(5, 6))
window.chart_canvas = FigureCanvasTkAgg(window.chart, window)
window.chart_canvas.get_tk_widget().grid(row=0,column=1)

frm_buttons = tk.Frame(window,relief=tk.RAISED,bd=2)
btn_open = tk.Button(frm_buttons,text="Выбрать изображение",command=open_image,state="normal")

btn_start = tk.Button(frm_buttons,text="Старт",command=start_work,state="disabled")
btn_open.grid(row=1,column=0,sticky="ew",padx=5,pady=0)
btn_start.grid(row=2,column=0,sticky="ew",padx=5,pady=5)
frm_buttons.grid(row=0,column=0,sticky="ns")


window.mainloop()



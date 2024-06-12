import tkinter as tk
from tkinter import simpledialog
import cv2 as cv
import os
import PIL.Image, PIL.ImageTk
import model
import camera

class App:

    def __init__(self, window=tk.Tk(), window_title="Sci-Fi Camera Classifier"):

        self.window = window
        self.window.title(window_title)

        self.counters = [1, 1]

        self.model = model.Model()

        self.auto_predict = False

        self.camera = camera.Camera()

        self.init_gui()

        self.delay = 15
        self.update()

        self.window.attributes("-topmost", True)
        self.window.mainloop()

    def init_gui(self):
        # Set up a futuristic-looking font
        font_style = ("Arial", 12, "bold")

        # Set up a sci-fi color scheme
        bg_color = "#151515"  # Dark background
        fg_color = "#00FF00"  # Green text
        button_bg = "#00FFFF"  # Cyan buttons
        button_fg = "#000000"  # Black text on buttons
        label_bg = "#000000"  # Black label background
        label_fg = "#FFFFFF"  # White label text

        # Set up the window background color
        self.window.config(bg=bg_color)

        # Set up the canvas for displaying the camera feed
        self.canvas = tk.Canvas(self.window, width=self.camera.width, height=self.camera.height, bg=bg_color)
        self.canvas.pack(side=tk.RIGHT, padx=20, pady=20)

        # Set up auto prediction toggle button
        self.btn_toggleauto = tk.Button(self.window, text="Auto Prediction", width=20, bg=button_bg, fg=button_fg, command=self.auto_predict_toggle)
        self.btn_toggleauto.pack(side=tk.TOP, padx=20, pady=20)

        # Set up the class name inputs
        self.classname_one = simpledialog.askstring("Classname One", "Enter the name of the first class:", parent=self.window)
        self.classname_two = simpledialog.askstring("Classname Two", "Enter the name of the second class:", parent=self.window)

        # Set up buttons for saving images for each class
        self.btn_class_one = tk.Button(self.window, text=self.classname_one, width=20, bg=button_bg, fg=button_fg, command=lambda: self.save_for_class(1))
        self.btn_class_one.pack(side=tk.TOP, padx=20, pady=20)

        self.btn_class_two = tk.Button(self.window, text=self.classname_two, width=20, bg=button_bg, fg=button_fg, command=lambda: self.save_for_class(2))
        self.btn_class_two.pack(side=tk.TOP, padx=20, pady=20)

        # Set up the train model button
        self.btn_train = tk.Button(self.window, text="Train Model", width=20, bg=button_bg, fg=button_fg, command=lambda: self.model.train_model(self.counters))
        self.btn_train.pack(side=tk.TOP, padx=20, pady=20)

        # Set up the predict button
        self.btn_predict = tk.Button(self.window, text="Predict", width=20, bg=button_bg, fg=button_fg, command=self.predict)
        self.btn_predict.pack(side=tk.TOP, padx=20, pady=20)

        # Set up the reset button
        self.btn_reset = tk.Button(self.window, text="Reset", width=20, bg=button_bg, fg=button_fg, command=self.reset)
        self.btn_reset.pack(side=tk.TOP, padx=20, pady=20)

        # Set up the class label
        self.class_label = tk.Label(self.window, text="CLASS", bg=label_bg, fg=label_fg, font=font_style)
        self.class_label.pack(side=tk.BOTTOM, padx=20, pady=20)

    def auto_predict_toggle(self):
        self.auto_predict = not self.auto_predict

    def save_for_class(self, class_num):
        ret, frame = self.camera.get_frame()
        if not os.path.exists("1"):
            os.mkdir("1")
        if not os.path.exists("2"):
            os.mkdir("2")

        cv.imwrite(f'{class_num}/frame{self.counters[class_num-1]}.jpg', cv.cvtColor(frame, cv.COLOR_RGB2GRAY))
        img = PIL.Image.open(f'{class_num}/frame{self.counters[class_num - 1]}.jpg')
        img.thumbnail((150, 150), PIL.Image.LANCZOS)
        img.save(f'{class_num}/frame{self.counters[class_num - 1]}.jpg')

        self.counters[class_num - 1] += 1

    def reset(self):
        for folder in ['1', '2']:
            for file in os.listdir(folder):
                file_path = os.path.join(folder, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)

        self.counters = [1, 1]
        self.model = model.Model()
        self.class_label.config(text="CLASS")

    def update(self):
        if self.auto_predict:
            print(self.predict())

        ret, frame = self.camera.get_frame()

        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        self.window.after(self.delay, self.update)

    def predict(self):
        ret, frame = self.camera.get_frame()
        if ret:
            prediction = self.model.predict(frame)
            if prediction == 1:
                self.class_label.config(text=self.classname_one)
                return self.classname_one
            elif prediction == 2:
                self.class_label.config(text=self.classname_two)
                return self.classname_two
            else:
                self.class_label.config(text="Unknown")
                return "Unknown"
        return None
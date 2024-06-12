from sklearn.svm import LinearSVC
import numpy as np
import cv2 as cv
import os

class Model:

    def __init__(self):
        self.model = LinearSVC()

    def load_and_preprocess_image(self, path):
        """Load and preprocess an image."""
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv.resize(img, (150, 150)).flatten()
        return img

    def collect_images(self, folder, label, counter):
        """Collect images and labels for a given folder and label."""
        img_list = []
        class_list = []
        for i in range(1, counter):
            img = self.load_and_preprocess_image(f'{folder}/frame{i}.jpg')
            if img is not None:
                img_list.append(img)
                class_list.append(label)
        return img_list, class_list

    def train_model(self, counters):
        img_list, class_list = [], []

        for folder, label, counter in zip(['1', '2'], [1, 2], counters):
            imgs, classes = self.collect_images(folder, label, counter)
            img_list.extend(imgs)
            class_list.extend(classes)

        if img_list and class_list:
            self.model.fit(np.array(img_list), np.array(class_list))
            print("Model successfully trained!")
        else:
            print("Training data is empty. Ensure you have collected images for both classes.")

    def preprocess_frame(self, frame):
        """Preprocess the input frame for prediction."""
        frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        frame = cv.resize(frame, (150, 150)).flatten()
        return frame

    def predict(self, frame):
        frame = self.preprocess_frame(frame)
        prediction = self.model.predict([frame])
        return prediction[0] if prediction else None
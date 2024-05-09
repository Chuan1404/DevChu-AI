from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from ultralytics import YOLO
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import cv2
import pandas as pd

model = YOLO("yolov8x-cls.pt")
index=["color","color_name","hex","R","G","B"]
csv = pd.read_csv('colors.csv', names=index, header=None)

COLORS = [
    (255, 0, 0),    # Đỏ
    (255, 165, 0),  # Cam
    (255, 255, 0),  # Vàng
    (0, 128, 0),    # Xanh lá
    (0, 0, 255),    # Xanh dương
    (75, 0, 130),   # Chàm
    (238, 130, 238),# Tím
    (255, 255, 255),# Trắng
    (0, 0, 0)       # Đen
]
class ClassificationModelAPIView(APIView):
    def get(self, request):
        return Response("ok", status=status.HTTP_200_OK)

    def post(self, request):

        file = request.FILES['file']
        image = Image.open(file)

        # classification image
        predict = self.classification(image)

        # process image
        image = self.process_image(image)

        # detect color
        color_detect = self.color_recognition(image)
        clusters = color_detect.get('names')

        color_detect['names'] = []
        color_detect['values'] = []
        for e in clusters:
            color_info = self.get_colorName(e)
            color_detect['names'].append(color_info.get('cname'))
            color_detect['values'].append(color_info.get('cvalue'))

        return Response({
            'predict': predict,
            'colors': color_detect
        }, status=status.HTTP_200_OK)

    def process_image(self, image):
        new_width = 224

        image = np.array(image)
        old_height, old_width = image.shape[:2]

        aspect_ratio = float(new_width) / old_width
        new_height = int(old_height * aspect_ratio)

        resized_img = cv2.resize(image, (new_width, new_height))

        return resized_img

    def classification(self, image):
        result = model(image)
        names = []
        probs = []

        i = 0
        for index in result[0].probs.top5:
            current_probs = np.round(result[0].probs.top5conf[i].item(), 2)
            if current_probs >= 0.5:
                names.append(result[0].names[index])
                probs.append(current_probs)
            i = i + 1

        index = result[0].probs.top1
        if names.__len__() == 0:
            names.append(result[0].names[index])
            probs.append(np.round(result[0].probs.top1conf.item(), 2))
        return {
            'names': names,
            'probs': probs
        }

    def color_recognition(self, image_file):
        image = np.array(image_file) / 255
        pixels = image.reshape((-1, 3))

        # Define the number of clusters (classes) you want
        k = 9

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(pixels)

        labels = kmeans.labels_
        cluster_sizes = np.bincount(labels, minlength=k)
        statistic = np.round(cluster_sizes / len(labels), 2)

        results = np.where(statistic >= 0.2)[0]
        if results.__len__() == 0:
            results = [np.argmax(statistic)]

        cluster_centers = np.uint8(kmeans.cluster_centers_ * 255)

        return {
            "names": cluster_centers[results],
            "probs": statistic[results],
        }

    def get_colorName(seft, rgb):
        [R, G, B] = rgb
        minimum = 10000
        for i in range(len(csv)):
            d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))

            if (d <= minimum):
                minimum = d
                cname = csv.loc[i, "color_name"]
                cvalue = csv.loc[i, "hex"]
        return {
            "cname": cname,
            "cvalue": cvalue,
        }

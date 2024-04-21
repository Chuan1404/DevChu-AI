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

        predict = self.classification(image)
        color_detect = self.color_recognition(image)

        clusters = color_detect.get('clusters')
        color_detect['clusters'] = [self.get_colorName(e) for e in clusters]
        # colorname = self.get_colorName(color_detect)
        # print(colorname)

        return Response({
            'predict': predict,
            'colors': color_detect
        }, status=status.HTTP_200_OK)

    def classification(self, image):
        result = model(image)
        names = []
        probs = []

        i = 0
        for index in result[0].probs.top5:
            current_probs = result[0].probs.top5conf[i].item()
            if current_probs >= 0.4:
                names.append(result[0].names[index])
                probs.append(current_probs)
            i = i + 1

        return {
            'names': names,
            'probs': probs
        }

    def color_recognition(self, image_file):
        # image = cv2.cvtColor(np.array(image_file), cv2.COLOR_RGB2BGR)
        image = np.array(image_file) / 255
        # Reshape the image into a 2D array of pixels
        pixels = image.reshape((-1, 3))

        # Define the number of clusters (classes) you want
        k = 9

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(pixels)

        labels = kmeans.labels_
        cluster_sizes = np.bincount(labels, minlength=k)
        statistic = cluster_sizes * 100 / len(labels)

        results = np.where(statistic >= 20)[0]
        if results.__len__ == 0:
            results = [np.argmax(statistic)]

        cluster_centers = np.uint8(kmeans.cluster_centers_ * 255)

        return {
            "clusters": cluster_centers[results],
            "probs": statistic[results]
        }

    def get_colorName(seft, rgb):
        [R, G, B] = rgb
        minimum = 10000
        for i in range(len(csv)):
            d = abs(R - int(csv.loc[i, "R"])) + abs(G - int(csv.loc[i, "G"])) + abs(B - int(csv.loc[i, "B"]))
            if (d <= minimum):
                minimum = d
                cname = csv.loc[i, "color_name"]
        return cname

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from ultralytics import YOLO
from PIL import Image

model = YOLO("yolov8x-cls.pt")


class ClassificationModelAPIView(APIView):

    def get(self, request):

        print(model.names)
        return Response("ok", status=status.HTTP_200_OK)

    def post(self, request):
        file = request.POST['file']
        print(file)

        result = model(file)
        names = []

        i = 0
        for index in result[0].probs.top5:
            names.append(result[0].names[index])
            i = i + 1
        return Response({
            'names': names,
            'probs': result[0].probs.top5conf
        }, status=status.HTTP_200_OK)

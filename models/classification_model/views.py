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
        file = request.FILES['file']
        image = Image.open(file)
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

        return Response({
            'names': names,
            'probs': probs
        }, status=status.HTTP_200_OK)

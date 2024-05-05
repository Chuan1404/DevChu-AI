from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from PIL import Image
import numpy as np
from stegano import lsb
from io import BytesIO
import cv2

class CopyRightAPIView(APIView):
    def get(self, request):
        return Response("ok", status=status.HTTP_200_OK)

    def post(self, request):

        # Đọc hình ảnh gốc và chữ ký
        file = request.FILES['file']
        original_image = Image.open(file)
        image = Image.open("signature.png")

        background = Image.new("RGB", image.size, (255, 255, 255))

        # Composite the original image onto the white background
        background.paste(image, mask=image.split()[3])

        # Convert the composite image to 24-bit depth (without alpha channel)
        signature = background.convert("RGB")

        # Chuyển đổi hình ảnh sang mảng NumPy
        original_pixels = np.array(original_image)
        red_channel_pixels = original_pixels[:, :, 0]
        signature_pixels = np.array(signature)

        # Lấy kích thước của ảnh gốc và chữ ký
        original_height, original_width = red_channel_pixels.shape
        signature_height, signature_width = signature_pixels.shape[:2]

        # Tạo mặt nạ chữ ký bằng cách tạo một ảnh trắng với kích thước tương tự
        signature_mask = np.full((original_height, original_width), 255, dtype=np.uint8)

        # Tạo mặt nạ
        num_repeats_h = original_height // signature_height
        num_repeats_w = original_width // signature_width
        for i in range(0, num_repeats_h + 1):
            for j in range(0, num_repeats_w + 1):
                y = i * signature_height
                x = j * signature_width
                end_y = min(y + signature_height, original_height)
                end_x = min(x + signature_width, original_width)

                crop_signature = signature_pixels[:end_y - y, :end_x - x, :]
                gray_crop_signature = np.mean(crop_signature, axis=2, dtype=np.uint8)  # Chuyển thành grayscale
                signature_mask[y:end_y, x:end_x] = gray_crop_signature
        # LSB
        signature_mask = (signature_mask > 0).astype(np.uint8)
        result = np.where(signature_mask & 1, red_channel_pixels | 1, red_channel_pixels & ~1)

        original_pixels[:, :, 0] = result

        # Tạo ảnh mới từ mảng NumPy đã sửa đổi
        modified_image = Image.fromarray(original_pixels)

        # Lưu ảnh đã ẩn chữ ký
        modified_image.save("hidden.jpg", quality=100, subsampling=0)
        return Response("ok", status=status.HTTP_200_OK)

class ExtractCopyRightAPIView(APIView):
    def get(self, request):
        return Response("ok", status=status.HTTP_200_OK)

    def post(self, request):
        file = request.FILES['file']
        # Load hình ảnh đã ẩn chữ ký
        image = Image.open(file)
        image_pixels = np.array(image)
        red_channel_pixels = image_pixels[:, :, 0]

        # Trích xuất bit cuối từ mỗi pixel trong kênh màu đỏ
        last_bits = red_channel_pixels & 1

        # Chuyển extracted_signature thành mảng NumPy và reshape lại thành kích thước ban đầu của ảnh
        extracted_signature = (last_bits * 255).astype(np.uint8)

        # Tạo ảnh chữ ký trích xuất từ mảng NumPy
        extracted_signature_image = Image.fromarray(extracted_signature)
        extracted_signature_image.save("extracted_signature.png")
        return Response("ok", status=status.HTTP_200_OK)

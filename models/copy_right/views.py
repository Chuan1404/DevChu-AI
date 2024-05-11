from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from PIL import Image
import numpy as np
from io import BytesIO
from django.http import HttpResponse
import os

class CopyRightAPIView(APIView):
    def get(self, request):
        return Response("ok", status=status.HTTP_200_OK)

    def post(self, request):

        if not request.FILES:
            return Response({
                "error": "File is not allowed null"
            }, status=status.HTTP_400_BAD_REQUEST)

        file = request.FILES['file']

        original_image = Image.open(file)
        file_name, file_extension = os.path.splitext(file.name)

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

        # Create a BytesIO object to hold the image data
        image_buffer = BytesIO()

        # Save the modified image to the BytesIO buffer
        file_upper = file_extension[1:].upper()
        if file_upper == "JPG":
            file_upper = "JPEG"
        Image.fromarray(original_pixels).save(image_buffer, format=file_upper, quality=100, subsampling=0)

        # Get the content of the BytesIO buffer
        image_data = image_buffer.getvalue()

        # Create an HTTP response with the image data
        response = HttpResponse(image_data, content_type='image/' + file_extension[1:])
        response['Content-Disposition'] = f'attachment; filename="{file.name}"'
        return response

class ExtractCopyRightAPIView(APIView):
    def get(self, request):
        return Response("ok", status=status.HTTP_200_OK)

    def post(self, request):

        if not request.FILES:
            return Response({
                "error": "File is not allowed null"
            }, status=status.HTTP_400_BAD_REQUEST)

        file = request.FILES['file']
        # Load hình ảnh đã ẩn chữ ký
        image = Image.open(file)
        image_pixels = np.array(image)
        red_channel_pixels = image_pixels[:, :, 0]

        # Trích xuất bit cuối từ mỗi pixel trong kênh màu đỏ
        last_bits = red_channel_pixels & 1

        # Chuyển extracted_signature thành mảng NumPy và reshape lại thành kích thước ban đầu của ảnh
        extracted_signature = (last_bits * 255).astype(np.uint8)

        image_buffer = BytesIO()

        Image.fromarray(extracted_signature).save(image_buffer, format="PNG")
        image_data = image_buffer.getvalue()
        response = HttpResponse(image_data, content_type='image/png')
        response['Content-Disposition'] = f'attachment; filename="extracted_signature.png"'
        return response

# class CopyRightAPIView(APIView):
#     def post(self, request):
#         file = request.FILES.get('file')
#         image = Image.open(file)
#         signature_image = Image.open("signature.png")
#         background = Image.new("RGB", signature_image.size, (255, 255, 255))
#
#         # Composite the original image onto the white background
#         background.paste(signature_image, mask=signature_image.split()[3])
#
#         # Convert the composite image to 24-bit depth (without alpha channel)
#         signature = background.convert("RGB")
#
#         # Chuyển đổi hình ảnh watermark thành grayscale
#         watermark_gray = signature.convert('L')
#         watermark_array = np.array(watermark_gray)
#
#         # Chia ảnh gốc thành các khối 8x8
#         image_array = np.array(image)
#         height, width, channels = image_array.shape
#
#         for i in range(0, height, 8):
#             for j in range(0, width, 8):
#                 block = image_array[i:i + 8, j:j + 8, 0]  # Chỉ lấy một kênh (ví dụ: kênh đỏ)
#                 dct_block = dct2(block)
#                 watermark = watermark_array[i // 8, j // 8] if i // 8 < watermark_array.shape[0] and j // 8 < \
#                                                                watermark_array.shape[1] else 0
#                 dct_block = hide_watermark_dct(dct_block, watermark)
#                 watermarked_block = idct2(dct_block)
#                 image_array[i:i + 8, j:j + 8, 0] = watermarked_block
#
#         watermarked_image = Image.fromarray(image_array.astype(np.uint8))  # Chuyển về kiểu dữ liệu uint8
#         watermarked_image.save('watermarked_image.jpg', quality=100, subsampling=0)
#
#         return Response({"message": "Watermark embedded successfully", "url": "watermarked_image.jpg"}, status=status.HTTP_200_OK)
#
# class ExtractCopyRightAPIView(APIView):
#     def post(self, request):
#         file = request.FILES.get('file')
#         watermarked_image = Image.open(file)
#
#         # Chia ảnh thành các khối 8x8
#         watermarked_array = np.array(watermarked_image)
#         height, width, channels = watermarked_array.shape
#         extracted_watermark_array = np.zeros((height // 8, width // 8), dtype=np.uint8)
#
#         for i in range(0, height, 8):
#             for j in range(0, width, 8):
#                 block = watermarked_array[i:i + 8, j:j + 8, 0]  # Chỉ lấy một kênh (ví dụ: kênh đỏ)
#                 watermark = extract_watermark_dct(block)
#                 extracted_watermark_array[i // 8, j // 8] = watermark
#
#         extracted_watermark_image = Image.fromarray(extracted_watermark_array)
#         extracted_watermark_image.save('extracted_watermark.jpg', quality=100, subsampling=0)
#         return Response({"message": "Watermark extracted successfully", "url": "extracted_watermark.jpg"}, status=status.HTTP_200_OK)
#
#
# # Hàm thực hiện DCT
# from scipy.fftpack import dct, idct
#
#
# def dct2(block):
#     return dct(block.T, norm='ortho').T
#
#
# # Hàm thực hiện ngược DCT
# def idct2(block):
#     return idct(block.T, norm='ortho').T
#
#
# # Hàm ẩn watermark vào hệ số DCT
# def hide_watermark_dct(block, watermark):
#     block[0, 0] += 0.1 * watermark   # Ẩn watermark vào hệ số DCT đầu tiên của khối
#     return block
#
# # Hàm rút trích thông tin từ hệ số DCT
# def extract_watermark_dct(block):
#     return (block[0, 0] / 0.1).astype(np.uint8)
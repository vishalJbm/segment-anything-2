from paddleocr import PaddleOCR, draw_ocr
import cv2
from matplotlib import pyplot as plt

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Use English model, adjust 'lang' for other languages

# Path to the image
img_path = 'WhatsApp Image 2024-08-29 at 4.33.40 PM.jpeg'

# Perform OCR on the image
result = ocr.ocr(img_path, cls=True)
print("===== :   ",result[-1][0])
# Display the results
for line in result:
    #print(line)
    pass

# # To visualize the results, load the image using OpenCV
# image = cv2.imread(img_path)

# # Extract text, bounding boxes, and recognition results
# boxes = [res[0] for res in result]
# texts = [res[1][0] for res in result]
# scores = [res[1][1] for res in result]

# # Draw results on the image
# image_with_boxes = draw_ocr(image, boxes, texts, scores, font_path='path_to_font.ttf')

# # Convert the image to RGB (from BGR) for plotting
# image_with_boxes = cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB)
# print("===== :   ",texts)
# # Plot the image with OCR results


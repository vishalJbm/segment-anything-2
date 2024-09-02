import matplotlib.pyplot as plt
import cv2,os

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from paddleocr import PaddleOCR, draw_ocr
import cv2
from matplotlib import pyplot as plt
from PIL import ImageFont


# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
ocr = PaddleOCR(use_angle_cls=True, lang='en')
if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()    

# Function to crop images with offset
def crop_with_offset(image, mask, offset=50):
    # Find bounding box of the mask
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
    
    # Calculate the crop area with the offset
    x1 = max(0, x - offset)
    y1 = max(0, y - offset)
    x2 = min(image.shape[1], x + w + offset)
    y2 = min(image.shape[0], y + h + offset)
    
    # Crop the image and mask
    cropped_image = image[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]
    
    return cropped_image, cropped_mask

def readChar(image_path):
    # Example font download
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    if not os.path.exists(font_path):
        print(f"Font not found at {font_path}, please provide a valid font path.")

    # Proceed with OCR as before
    
    # image_path = "/home/reemaganotra/Shyam/OCR/segment-anything-2/notebooks/images_segmented/WhatsApp Image 2024-08-28 at 17.27.21.jpeg"
    result = ocr.ocr(image_path, cls=True)

    image = cv2.imread(image_path)
    boxes, texts, scores = [], [], []
    for line in result:
        for res in line:
            if len(res) > 1 and len(res[1]) == 2:
                boxes.append(res[0])
                texts.append(res[1][0])
                scores.append(res[1][1])

    if boxes and texts and scores:
        image_with_boxes = draw_ocr(image, boxes, texts, scores, font_path=font_path)
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
        plt.show()
    else:
        print("No text detected or extraction failed.")

sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")

predictor = SAM2ImagePredictor(sam2_model)
# print("--predictor--"*10,predictor)

h_,w_  = 1685,1685
import os
read_path = '/home/reemaganotra/Shyam/OCR/segment-anything-2/notebooks/images/c4'
write_path = '/home/reemaganotra/Shyam/OCR/segment-anything-2/notebooks/images_segmented'

cam_id = [ 'c4_top']
offset = 200

for cam in cam_id:
   
    # if cam =='c4_top' :
    read_path = read_path#.replace(read_path.split('/')[-1], 'c4')

    for im in os.listdir(read_path):
        input_path = os.path.join(read_path,im)
        print(input_path, ">>>>>>>>>>")
        
        # input_path = os.path.join(read_path,im)
        output_p = write_path
        if not os.path.exists(output_p):
            os.makedirs(output_p)
        output_path = os.path.join(output_p,im)

        print(output_path,"output")
        image = cv2.imread(input_path)
        center = image.shape
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        predictor.set_image(image)
        input_label = np.array([1])
        
        if cam == 'c4_top':
            # box = [706,900,1036,1305] #c4_top 
            box = [904,405,1009,566] #c4_top 
            input_point =  np.array([[974, 506]])
            # box = [658,755,745,836] #c4_top 
            # input_point =  np.array([[707, 790]])
            
            multimask_output=  False
       
        masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=multimask_output,
        box = box
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
        # show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)

            # Apply cropping to each mask
        cropped_images = []
        cropped_masks = []
        for i in range(len(masks)):
            cropped_image, cropped_mask = crop_with_offset(image, masks[i])
            cropped_images.append(cropped_image)
            cv2.imwrite(output_path, cropped_image)
            readChar(output_path)
            cropped_masks.append(cropped_mask)

    # for i in range(len(cropped_masks)):
    #     show_masks(cropped_images[i], [cropped_masks[i]], [scores[i]], point_coords=input_point, input_labels=input_label, borders=True)




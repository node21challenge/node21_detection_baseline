import SimpleITK as sitk
import json
import numpy as np
from PIL import Image, ImageDraw
import os
from PIL import ImageFont

path_name = "/Volumes/ExternalHardDrive/node21_detection_baseline/Untitled/input/"
images = []
for filename in os.listdir(path_name):
    full_path = path_name + filename
    print(full_path)
    image = sitk.ReadImage(full_path, imageIO="MetaImageIO")
    image_arr = sitk.GetArrayFromImage(image)
    max_arr = np.max(image_arr)
    image_arr = image_arr / max_arr
    image_arr = image_arr * 255
    img = Image.fromarray(image_arr).convert('RGB')
    images.append(img)

# plt.imshow(image_arr)
with open('/Volumes/ExternalHardDrive/node21_detection_baseline/Untitled/output/nodules.json', 'r') as f:
    bounding_boxes = json.load(f)

points = bounding_boxes['boxes']

for k,img in enumerate(images):
    # create rectangle image
    point = points[k]
    for i, point in enumerate(points):
        probability = point['probability']
        img1 = ImageDraw.Draw(img)
        if probability > 0.2:
            corner_pts = point['corners']
            text_pos = (int(corner_pts[0]), int(corner_pts[1] - 30))
            font = ImageFont.truetype("/Library/Fonts/Arial.ttf", size=24)
            if i == 0:
                img1.rectangle(corner_pts, outline="blue")
                img1.text(
                    text_pos,  # Coordinates
                    str(probability),  # Text
                    (0, 0, 255),  # Color
                    font=font
                )
            else:
                img1.rectangle(corner_pts, outline="red")
                img1.text(
                    text_pos,  # Coordinates
                    str(probability),  # Text
                    (255, 0, 0),  # Color
                    font=font
                )
    img.show()

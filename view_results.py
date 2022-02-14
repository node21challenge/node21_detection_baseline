import SimpleITK as sitk
import json
import numpy as np
from PIL import Image, ImageDraw
import os
from PIL import ImageFont

path_name = "/Users/nikki/Desktop/node21_detection_baseline/input/"
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
with open('/Users/nikki/Desktop/node21_detection_baseline/output/nodules.json', 'r') as f:
    bounding_boxes = json.load(f)

points = bounding_boxes['boxes']

for i,img in enumerate(images):
    # create rectangle image
    point = points[i]
    for i, point in enumerate(points):
        probability = point['probability']
        img1 = ImageDraw.Draw(img)

        corner_pts = point['corners']
        text_pos = (int(corner_pts[0]), int(corner_pts[1] - 30))
        font = ImageFont.truetype("/Library/Fonts/Arial.ttf", size=24)
        img1.text(
          text_pos,  # Coordinates
          str(probability),  # Text
          (255, 0, 0),  # Color
          font=font
        )
        img1.rectangle(corner_pts, outline="red")
    img.show()

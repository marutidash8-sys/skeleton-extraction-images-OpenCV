import cv2
import numpy as np
import os

# Input folder
input_folder = "data/"
output_folder = "output/"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def skeletonize(image):
    size = np.size(image)
    skel = np.zeros(image.shape, np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(image, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(image, temp)
        skel = cv2.bitwise_or(skel, temp)
        image = eroded.copy()

        zeros = size - cv2.countNonZero(image)
        if zeros == size:
            done = True

    return skel

# Process all images in data folder
for file in os.listdir(input_folder):
    if file.lower().endswith(("jpg", "png", "jpeg")):
        img_path = os.path.join(input_folder, file)
        img = cv2.imread(img_path, 0)

        # Pre-processing
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Skeleton extraction
        skeleton = skeletonize(thresh)

        # Save output
        output_file = os.path.join(output_folder, file.split('.')[0] + "_skeleton.png")
        cv2.imwrite(output_file, skeleton)

        print(f"Skeleton saved: {output_file}")

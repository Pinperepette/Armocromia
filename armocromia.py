#!/usr/bin/env python
import cv2
import numpy as np
from sklearn.cluster import KMeans
import json
from skimage.color import rgb2lab, deltaE_cie76
from PIL import Image
import io
import subprocess
import sys
import os
from skimage.color import deltaE_ciede2000

def import_image(file_path):
    try:
        pil_image = Image.open(file_path)
        image = np.array(pil_image)

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception as e:
        print(f"Error: Could not read image from {file_path}: {e}")
        return None

def resize_image(image, height=600, length=400):
    return cv2.resize(image, (length, height), interpolation=cv2.INTER_AREA)

def random_sample(image, sample_size=1000):
    h, w, _ = image.shape
    indices = np.random.choice(h * w, sample_size)
    sampled_pixels = image.reshape(-1, 3)[indices]
    return sampled_pixels

def get_main_color(image, k=5, sample_size=1000):
    sampled_pixels = random_sample(image, sample_size)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(sampled_pixels)
    colors_principali = kmeans.cluster_centers_
    return colors_principali

def armocromia_adatta(image):
    colors = get_main_color(image)
    armocromie = {
        "autunno": [178, 124, 87],
        "inverno": [80, 70, 100],
        "primavera": [246, 206, 163],
        "estate": [196, 180, 188]
    }
    distanze = {}
    for armocromia, color in armocromie.items():
        color_rgb = np.array(color).reshape(1, 1, 3).astype(np.float64) 
        color_lab = rgb2lab(color_rgb)[0][0]
        colors_principali_rgb = colors.reshape(-1, 1, 3).astype(np.float64)
        colors_principali_lab = rgb2lab(colors_principali_rgb).reshape(-1, 3)
        distanze[armocromia] = np.mean([deltaE_ciede2000(color_lab, c_lab) for c_lab in colors_principali_lab])
    armocromia_adatta = min(distanze, key=distanze.get)
    return armocromia_adatta

def show_image(image, titolo=''):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    with io.BytesIO() as output:
        pil_image.save(output, format="PNG")
        img_data = output.getvalue()
    with open("temp_image.png", "wb") as f:
        f.write(img_data)
    term = os.getenv('TERM')
    if term == 'xterm-256color':
        subprocess.run(["img2sixel", "temp_image.png"])
    else:
        subprocess.run(["open", "temp_image.png"])

def show_palette(armocromia):
    palette_colors = {
        "autunno": [
            [120, 65, 45],
            [190, 140, 110],
            [240, 180, 140],
            [170, 110, 50],
            [255, 215, 180]
        ],
        "inverno": [
            [40, 45, 60],
            [150, 150, 170],
            [250, 250, 255],
            [120, 130, 140],
            [200, 200, 220]
        ],
        "primavera": [
            [250, 230, 210],
            [240, 200, 160],
            [230, 160, 110],
            [255, 240, 225],
            [255, 220, 200]
        ],
        "estate": [
            [190, 190, 200],
            [230, 230, 240],
            [160, 160, 180],
            [200, 200, 220],
            [250, 250, 255]
        ]
    }

    palette = np.zeros((50, 250, 3), dtype=np.uint8)
    colors = palette_colors[armocromia]

    for i, color in enumerate(colors):
        palette[:, i * 50:(i + 1) * 50, :] = color

    pil_palette = Image.fromarray(palette)
    with io.BytesIO() as output:
        pil_palette.save(output, format="PNG")
        palette_data = output.getvalue()
    with open("temp_palette.png", "wb") as f:
        f.write(palette_data)
    subprocess.run(["img2sixel", "temp_palette.png"])

if __name__ == "__main__":
    file_path = sys.argv[1]
    image = import_image(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    armocromia = armocromia_adatta(image)
    print(f"The most suitable color analysis for the image {file_path} is: {armocromia}\n")
    show_image(image)
    show_palette(armocromia)

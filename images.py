import tkinter as tk
from tkinter import Entry, PhotoImage, Tk, filedialog, messagebox, ttk
from tkinter.ttk import Button, Label

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
from skimage import data, feature
from skimage.color import rgb2gray
from skimage.feature import ORB, blob_doh, match_descriptors, plot_matches
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, rotate

pathIn = ''
interface = Tk()
xPosition = ttk.Entry(interface)
yPosition = ttk.Entry(interface)
scale = ttk.Entry(interface)
angle = ttk.Entry(interface)
matches_array = []


def openFile():
    global pathIn
    imagenPath = filedialog.askopenfilename(
        title="Seleccionar archivo de video", filetypes=[("Image File", '.jpg'), ("Image File", '.png'), ("Image File", '.jpeg')])
    pathIn = imagenPath
    if pathIn == "":
        messagebox.showwarning("¡ERROR!", "Debes seleccionar una imagen")


def moveImage():
    global pathIn
    img = skimage.io.imread(pathIn)
    num_rows, num_cols = img.shape[:2]

    translation_matrix = np.float32(
        [[1, 0, int(xPosition.get())], [0, 1, int(yPosition.get())]])
    img_translation = cv2.warpAffine(
        img, translation_matrix, (num_cols, num_rows))
    mainProcess(img_translation)


def resizeImage():
    global pathIn
    img = skimage.io.imread(pathIn)
    print('Original Dimensions : ', img.shape)

    scale_percent = int(scale.get())  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    print('Resized Dimensions : ', resized.shape)
    mainProcess(resized)


def rotateImage():
    global pathIn
    matches_array = []
    grades_array = []
    coincidence = []
    img = skimage.io.imread(pathIn)
    for i in range(0, 360, int(angle.get())):
        grades_array.append(i)
        image_center = rotate(img, i, resize=True)
        matches, keypoints = mainProcess(image_center)
        coincidence.append(len(params(matches, keypoints)))
        matches_array.append((len(matches)/len(keypoints))*100)
    print(grades_array, matches_array, coincidence)
    plt.title('DoH')
    plt.xlabel('Grados de transformación')
    plt.ylabel('Porcentaje (%)')
    plt.plot(grades_array, matches_array,
             color="blue", label="No. coincidencias")
    plt.xticks(grades_array)
    plt.legend()
    plt.show()


def procesar():
    global pathIn
    image = skimage.io.imread(pathIn)
    image_gray = rgb2gray(image)

    blobs_doh = blob_doh(image_gray, min_sigma=1, max_sigma=30,
                         num_sigma=10, threshold=0.01, overlap=0.5, log_scale=False)
    fig, grafica = plt.subplots(1)

    grafica.set_title("DoH")
    grafica.imshow(image)

    for blob in blobs_doh:
        y, x, r = blob
        c = plt.Circle((x, y), r, color="red", linewidth=2, fill=False)
        grafica.add_patch(c)
    grafica.set_axis_off()
    plt.show()


def mainProcess(image):
    global pathIn

    image_gray = rgb2gray(image)
    img_path_original = skimage.io.imread(pathIn)
    img_original = rgb2gray(img_path_original)

    blobs_doh = blob_doh(img_original, min_sigma=1, max_sigma=30,
                         num_sigma=10, threshold=0.01, overlap=0.5, log_scale=False)

    descriptor_extractor = ORB(n_keypoints=len(blobs_doh))
    descriptor_extractor.detect_and_extract(img_original)
    keypoints1 = descriptor_extractor.keypoints
    descriptors1 = descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(image_gray)
    keypoints2 = descriptor_extractor.keypoints
    descriptors2 = descriptor_extractor.descriptors

    matches = match_descriptors(descriptors1, descriptors2, cross_check=True)

    return matches, keypoints1

    # fig, ax = plt.subplots(nrows=2, ncols=1)

    # plot_matches(ax[0], img_path_original, image,
    #              keypoints1, keypoints2, matches)

    # ax[0].axis('off')
    # ax[0].set_title('DoH')

    # ax[1].set_title('Número de coincidencias')
    # ax[1].plot(matches_array)
    # ax[1].set(xlabel='Paso de transformación', ylabel='%')

    # plt.show()


def params(matches, keypoints):
    matches_list = []
    for i in matches:
        result = ((len(matches)*100))/len(keypoints)
        # result = (i/len(keypoints))*100
        matches_list.append(result)
    return matches_list


photo = PhotoImage(file=r"btn.png")

Button(interface, text='Seleccionar imagen', image=photo,
       command=openFile).grid(row=1, column=1)

Label(interface, text="X").grid(row=1, column=2)
xPosition.grid(row=1, column=3)

Label(interface, text="Y").grid(row=2, column=2)
yPosition.grid(row=2, column=3)

Button(interface, text='Procesar imagen movida',
       command=moveImage).grid(row=3, column=3)

Label(interface, text="Escalado").grid(row=1, column=4)
scale.grid(row=1, column=5)

Button(interface, text="Procesar imagen con escalado",
       command=resizeImage).grid(row=2, column=5)

Label(interface, text='Grados').grid(row=1, column=6)
angle.grid(row=1, column=7)

Button(interface, text='Procesar imagen orientada',
       command=rotateImage).grid(row=2, column=7)

Button(interface, text="Procesar imagen (default)",
       command=procesar).grid(row=1, column=8)
interface.mainloop()

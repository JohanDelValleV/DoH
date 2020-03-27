import tkinter as tk
from tkinter import Entry, PhotoImage, Tk, filedialog, messagebox, ttk
from tkinter.ttk import Button, Label
import math
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
    matches_array = []
    coincidence = []
    position_x = int(xPosition.get())
    positions = [(0, position_x), (position_x, position_x), (position_x, 0), (position_x, -position_x),
                 (0, -position_x), (-position_x, -position_x), (-position_x, 0), (-position_x, position_x)]
    img = skimage.io.imread(pathIn)
    moved_cords = []

    num_rows, num_cols = img.shape[:2]
    for i, z in positions:
        translation_matrix = np.float32(
            [[1, 0, i], [0, 1, z]])
        img_translation = cv2.warpAffine(
            img, translation_matrix, (num_cols + 110, num_rows + 110))

        matches, keypoints_original, keypoints_transformada = mainProcess(
            img_translation)
        for blobs in keypoints_original:
            y, x, r = blobs
            moved_cords.append((x+z, y+i))

        compare_coords(keypoints_transformada, moved_cords)
        moved_cords.clear()

        coincidence.append(len(params(matches, keypoints_original)))
        matches_array.append((len(matches)/len(keypoints_original))*100)
    # print(matches_array, coincidence)

    plt.title('Transformación de imagen (DoH)')
    plt.xlabel('Transformación')
    plt.ylabel('Porcentaje (%)')
    plt.plot(matches_array,
             color="blue", label="No. coincidencias")
    plt.legend()
    plt.show()


def compare_coords(keypoints, new_coords):
    keypoints_x_y = []
    matches = []
    for blobs in keypoints:
        y, x, r = blobs
        keypoints_x_y.append((x, y))

    for kp_cords in [list(a) for a in keypoints_x_y]:
        best_match = 5
        for kp_trans in [list(b) for b in new_coords]:
            result = math.sqrt(
                pow((kp_trans[0]-kp_cords[0]), 2) + pow((kp_trans[1]-kp_cords[1]), 2))
            if result < best_match:
                best_match = result
        if best_match <= 3 or best_match >= 3:
            matches.append(best_match)
    print(f'Matches: {matches}')


def resizeImage():
    global pathIn
    coincidence = []
    size_array = [400, 200, 50, 25]
    matches_array = []
    resized_cords = []
    img = cv2.imread(pathIn)

    for i in size_array:
        width = int(img.shape[1] * i / 100)
        height = int(img.shape[0] * i / 100)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        matches, keypoints_original, keypoints_transformada = mainProcess(
            resized)
        for blobs in keypoints_original:
            y, x, r = blobs
            resized_cords.append(((x*(i/100)), (y*(i/100))))
        print(f'Resized cords: {resized_cords}')
        # compare_coords(keypoints_transformada, resized_cords)

        resized_cords.clear()

        coincidence.append(len(params(matches, keypoints_original)))
        matches_array.append((len(matches)/len(keypoints_original))*100)
    # print(size_array, matches_array, coincidence)

    plt.title('Escalado de imagen (DoH)')
    plt.xlabel('Tamaño de transformación')
    plt.ylabel('Porcentaje (%)')
    plt.plot(size_array, matches_array,
             color="blue", label="No. coincidencias")
    plt.xticks(size_array)
    plt.legend()
    plt.show()


def rotate_kp(p, origin=(0, 0), degrees=0):
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)


def rotate_around_point_lowperf(point, radians, origin=(0, 0)):
    x, y = point
    ox, oy = origin

    qx = ox + math.cos(radians) * (x - ox) + math.sin(radians) * (y - oy)
    qy = oy + -math.sin(radians) * (x - ox) + math.cos(radians) * (y - oy)

    return qx, qy


def rotateImage():
    global pathIn
    matches_array = []
    grades_array = []
    coincidence = []
    coords = []
    img = skimage.io.imread(pathIn)
    num_rows_original, num_cols_original = img.shape[:2]
    num_rows_original = num_rows_original/2
    num_cols_original = num_cols_original/2

    for i in range(int(angle.get()), 360, int(angle.get())):
        grades_array.append(i)
        image_center = rotate(img, i, resize=True)
        num_rows_rotated, num_cols_rotated = image_center.shape[:2]
        num_rows_rotated = num_rows_rotated/2
        num_cols_rotated = num_cols_rotated/2

        cv2.imshow('Image', image_center)
        cv2.waitKey(0)

        matches, keypoints_original, keypoints_transformada = mainProcess(
            image_center)
        print(f'Angulo: {i}')
        for blobs in keypoints_original:
            y, x, r = blobs
            # print(rotate_kp((x, y),
            #                 origin=(num_rows_original+(num_rows_original - num_rows_rotated),
            #                         num_cols_original+(num_cols_original-num_cols_rotated)), degrees=i))
            print(rotate_around_point_lowperf((x, y), math.radians(i), origin=(num_cols_original+(num_cols_original - num_cols_rotated),
                                                                               num_rows_original+(num_rows_original-num_rows_rotated))))
        # coords.clear()
        coincidence.append(len(params(matches, keypoints_original)))
        matches_array.append((len(matches)/len(keypoints_original))*100)
    # print(grades_array, matches_array, coincidence)

    plt.title('Rotación de imagen (DoH)')
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
    # print(f'Imagen original: {blobs_doh}')

    blobs_doh2 = blob_doh(image_gray, min_sigma=1, max_sigma=30,
                          num_sigma=10, threshold=0.01, overlap=0.5, log_scale=False)
    print(f'Imagen transformada: {blobs_doh2}')

    descriptor_extractor = ORB(n_keypoints=len(blobs_doh))

    descriptor_extractor.detect_and_extract(img_original)
    # keypoints1 = descriptor_extractor.keypoints
    descriptors1 = descriptor_extractor.descriptors

    descriptor_extractor.detect_and_extract(image_gray)
    # keypoints2 = descriptor_extractor.keypoints
    descriptors2 = descriptor_extractor.descriptors

    # fig, ax = plt.subplots(1)
    matches = match_descriptors(
        descriptors1, descriptors2, cross_check=True, p=2, max_ratio=0.8)
    # plot_matches(ax, img_path_original, image, blobs_doh, blobs_doh2, matches)

    return matches, blobs_doh, blobs_doh2


def params(matches, keypoints):
    matches_list = []
    for i in matches:
        result = ((len(matches)*100))/len(keypoints)
        matches_list.append(result)
    return matches_list


photo = PhotoImage(file=r"btn.png")

Button(interface, text='Seleccionar imagen', image=photo,
       command=openFile).grid(row=1, column=1)

Label(interface, text='Rango de movimiento').grid(row=1, column=2)
xPosition.grid(row=1, column=3)

Button(interface, text='Procesar imagen (movimiento)',
       command=moveImage).grid(row=2, column=2)

Button(interface, text="Procesar imagen (escalado)",
       command=resizeImage).grid(row=2, column=5)

Label(interface, text='Grados').grid(row=1, column=6)
angle.grid(row=1, column=7)

Button(interface, text='Procesar imagen (orientada)',
       command=rotateImage).grid(row=2, column=7)

interface.mainloop()

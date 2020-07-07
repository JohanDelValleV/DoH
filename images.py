import tkinter as tk
from tkinter import Entry, PhotoImage, Tk, filedialog, messagebox, ttk
from tkinter.ttk import Button, Label
import math
import copy
import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
from skimage import data, feature
from skimage.color import rgb2gray
from skimage.feature import blob_doh
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, rotate
from matplotlib.patches import ConnectionPatch

pathIn = ''
interface = Tk()
xPosition = ttk.Entry(interface)
yPosition = ttk.Entry(interface)
scale = ttk.Entry(interface)
angle = ttk.Entry(interface)
matches_array = []
keypoints_original_blob = []
keypoints_transformed_blob = []


def openFile():
    global pathIn
    imagenPath = filedialog.askopenfilename(
        title="Seleccionar archivo de video", filetypes=[("Image File", '.jpg'), ("Image File", '.png'), ("Image File", '.jpeg')])
    pathIn = imagenPath
    if pathIn == "":
        messagebox.showwarning("¡ERROR!", "Debes seleccionar una imagen")


def moveImage():
    global pathIn
    position_x = int(xPosition.get())
    positions = [(0, position_x), (position_x, position_x), (position_x, 0), (position_x, -position_x),
                 (0, -position_x), (-position_x, -position_x), (-position_x, 0), (-position_x, position_x)]
    positions_in_directions = ['N', 'NE', 'E', 'SE', 'S', 'SO', 'O', 'NO']
    moved_cords = []
    img = skimage.io.imread(pathIn)
    num_rows_original, num_cols_original = img.shape[:2]

    img_origin = cv2.copyMakeBorder(
        img, position_x, position_x, position_x, position_x, cv2.BORDER_CONSTANT, value=None)

    new_num_rows, new_num_cols = img_origin.shape[:2]
    for i, z in positions:
        translation_matrix = np.float32(
            [[1, 0, i], [0, 1, z]])
        img_translation = cv2.warpAffine(
            img_origin, translation_matrix, (new_num_cols, new_num_rows))

        keypoints_original, keypoints_transformada = mainProcess(img_origin,
                                                                 img_translation)
        for blobs in keypoints_original:
            y, x, r = blobs
            moved_cords.append(((x+i, y+z), (x, y)))

        compare_coords(keypoints_transformada, keypoints_original,
                       moved_cords, img_origin, img_translation)
        moved_cords.clear()
    graph(positions_in_directions, matches_array)
    matches_array.clear()


def compare_coords(keypoints_transformed, keypoints_originales, kp_by_me, original_image, transformed_image):
    global matches_array
    keypoints_transformed_x_y = []
    keypoints_original_list = []
    matches = []

    for blobs in keypoints_transformed:
        y, x, r = blobs
        keypoints_transformed_x_y.append((x, y))

    for xy in keypoints_originales:
        y, x, r = xy
        keypoints_original_list.append((x, y))

    keypoints_transformed_list = [list(a) for a in keypoints_transformed_x_y]
    keypoints_by_me = [list(b) for b in kp_by_me]
    print(f'TRANSFORMED-------> {keypoints_transformed_list}')
    print(f'BY ME-------> {kp_by_me}\n')

    for kp_trans, kp_origin in keypoints_by_me:
        best_match = 10
        for kp_cords in keypoints_transformed_list:
            result = math.sqrt(
                pow((kp_trans[0]-kp_cords[0]), 2) + pow((kp_trans[1]-kp_cords[1]), 2))
            if result < best_match:
                best_match = result
        if best_match <= 4:
            matches.append((kp_trans, kp_origin))
    print(f'Matches: {matches}')
    draw_matches(keypoints_transformed_x_y, matches, keypoints_by_me,
                 original_image, transformed_image)
    matches_array.append(params(matches, keypoints_original_list))
    print(matches_array)


def graph(transformation, matches):
    global matches_array
    plt.title('DoH')
    plt.xlabel('Grados de transformación')
    plt.ylabel('Porcentaje(%) de coincidencia')
    plt.bar(transformation, matches_array, width=2, align='center', label='No. coincidencias')
    plt.xticks(transformation)
    plt.legend()
    plt.show()


def draw_matches(keypoints_transformed, matches, kp_by_me, original_image, transformed_image):
    global keypoints_original_blob
    keypoints_original_x_y = []
    img_copy = copy.copy(transformed_image)

    for blobs in keypoints_original_blob:
        y, x, r = blobs
        keypoints_original_x_y.append((x, y))

    for x in keypoints_original_x_y:
        cv2.circle(original_image, (int(x[0]), int(x[1])), 3, (255, 0, 0), -1)

    cv2.imshow('Imagen original', original_image)
    cv2.waitKey()

    for x2 in keypoints_transformed:
        cv2.circle(transformed_image,
                   (int(x2[0]), int(x2[1])), 3, (255, 255, 0), -1)

    cv2.imshow('Imagen desplazada', transformed_image)
    cv2.waitKey()

    for x3, x4 in kp_by_me:
        cv2.circle(img_copy, (int(x3[0]), int(x3[1])), 3, (0, 128, 0), -1)

    cv2.imshow('Imagen a kp_by_me', img_copy)
    cv2.waitKey()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(original_image)
    ax2.imshow(img_copy)
    ax1.set_axis_off()
    ax2.set_axis_off()
    coordsA = "data"
    coordsB = "data"

    for i, z in matches:
        con = ConnectionPatch(xyA=i, xyB=z, coordsA=coordsA, coordsB=coordsB,
                              axesA=ax2, axesB=ax1, arrowstyle="-", color="red")
        ax2.add_artist(con)
    # plt.tight_layout()
    plt.show()


def resizeImage():
    global pathIn
    size_array = [400, 200, 50, 25]
    resized_cords = []
    img = cv2.imread(pathIn)

    for i in size_array:
        width = int(img.shape[1] * i / 100)
        height = int(img.shape[0] * i / 100)
        dim = (width, height)
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        keypoints_original, keypoints_transformada = mainProcess(img,
                                                                 resized)
        for blobs in keypoints_original:
            y, x, r = blobs
            resized_cords.append((((x*(i/100)), (y*(i/100))), (x, y)))
        print(f'Resized cords: {resized_cords}')
        compare_coords(keypoints_transformada, keypoints_original,
                       resized_cords, img, resized)
        resized_cords.clear()
    graph(size_array, matches_array)
    matches_array.clear()


def rotate_kp(point, radians, origin=(0, 0)):
    x, y = point
    ox, oy = origin

    qx = ox + math.cos(radians) * (x - ox) + math.sin(radians) * (y - oy)
    qy = oy + -math.sin(radians) * (x - ox) + math.cos(radians) * (y - oy)

    return (qx, qy)


def rotateImage():
    global pathIn
    global matches_array
    grades_array = []
    kp_by_me = []

    img = skimage.io.imread(pathIn)
    num_rows_original, num_cols_original = img.shape[:2]

    hip = math.sqrt(pow(num_rows_original, 2) + pow(num_cols_original, 2))//2

    img_origin = cv2.copyMakeBorder(img, int(hip-(num_rows_original//2)), int(hip-(num_rows_original//2)), int(hip-(
        num_cols_original//2)), int(hip-(num_cols_original//2)), cv2.BORDER_CONSTANT, value=None)

    new_num_rows_original, new_num_cols_original = img_origin.shape[:2]
    new_num_cols_original = new_num_cols_original/2
    new_num_rows_original = new_num_rows_original/2

    for i in range(int(angle.get()), 360, int(angle.get())):
        grades_array.append(i)
        image_center = rotate(img_origin, i, resize=False)
        num_cols_rotated, num_rows_rotated = image_center.shape[:2]
        num_cols_rotated = num_cols_rotated/2
        num_rows_rotated = num_rows_rotated/2

        keypoints_original, keypoints_transformada = mainProcess(
            img_origin, image_center)
        print(f'Angulo: {i}')
        for blobs in keypoints_original:
            y, x, r = blobs
            kp_by_me.append((rotate_kp((x, y), math.radians(i), origin=(new_num_cols_original,
                                                                        new_num_rows_original)), (x, y)))
        print(f'Juntos: {kp_by_me}')
        compare_coords(keypoints_transformada,
                       keypoints_original, kp_by_me, img_origin, image_center)
        kp_by_me.clear()
    graph(grades_array, matches_array)
    matches_array.clear()


def mainProcess(image_original, image_transformed):
    global pathIn
    global keypoints_original_blob
    global keypoints_transformed_blob
    image_gray = rgb2gray(image_transformed)
    img_path_original = skimage.io.imread(pathIn)
    img_original = rgb2gray(image_original)

    blobs_doh = blob_doh(img_original, min_sigma=1, max_sigma=30,
                         num_sigma=10, threshold=0.01, overlap=0.5, log_scale=False)
    keypoints_original_blob = blobs_doh

    blobs_doh2 = blob_doh(image_gray, min_sigma=1, max_sigma=30,
                          num_sigma=10, threshold=0.01, overlap=0.5, log_scale=False)
    keypoints_transformed_blob = blobs_doh2

    return blobs_doh, blobs_doh2


def params(matches, keypoints):
    result = ((len(matches)*100))/len(keypoints)
    return result


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

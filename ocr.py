import numpy as np
import imutils
import cv2
import imgaug
import keras_ocr
import easyocr
import os
import matplotlib.pyplot as plt 
import pytesseract #OCR Tesseract
#import tensorflow as tf
reader = easyocr.Reader(['es'])
recognizer = keras_ocr.recognition.Recognizer()
recognizer.model.load_weights(os.path.abspath('weights/SPIID_v6_test.h5'))
recognizer.compile()



def resize_image(img, size=(300,300)):

    h, w = img.shape[:2]
    c = img.shape[2] if len(img.shape)>2 else 1

    if h == w:
        return cv2.resize(img, size, interpolation = cv2.INTER_AREA)

    dif = h if h > w else w

    interpolation = cv2.INTER_AREA if dif > (size[0]+size[1])//2 else cv2.INTER_CUBIC

    x_pos = (dif - w)//2
    y_pos = (dif - h)//2

    if len(img.shape) == 2:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype) + 255
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return cv2.resize(mask, size, interpolation)

def recognize_numbers(crop, psm=7, vote='hard',weights_keras='SPIID_v6_test.h5'):
    print(crop)
    list_num = [];
    list_conf = []
    image_or = cv2.imread(crop)
    image = imutils.resize(image_or, width=200)

    # Grayscale conversion
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray)

    # Smoothing before thresholding
    blurred = cv2.GaussianBlur(gray_clahe, (5, 5), 0)

    # Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Sharpening filter
    kernel = np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])
    sharp = cv2.filter2D(adaptive_thresh, -1, kernel)

    # Inverting the binary image for OCR
    roi = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # OCR detection
    d = reader.readtext(roi, contrast_ths=0.2, adjust_contrast=0.7, min_size=1, mag_ratio=1)

    #Original transformation
    #image = resize_image(image_or)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    #light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
    #light = cv2.threshold(light, 0, 255,
    #                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,
    #              dx=1, dy=0, ksize=-1)

    #roi = cv2.threshold(light, 0, 255,
    #                    cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    #d = reader.readtext(roi,  contrast_ths=0.2, adjust_contrast=0.7, min_size=1, mag_ratio=1)


    print(d)
    if d != []:
       # print('deblured', d)  # (poner subrutina)
        text, conf = d2text(d)
        list_num.append(text)
        list_conf.append(conf)

    if d == [] or conf < 0.9999:
        # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.bitwise_not(gray)
        smooth = cv2.GaussianBlur(gray, (5, 5), 0)

        d = reader.readtext(smooth, contrast_ths=0.2, adjust_contrast=0.7, min_size=1,
                            mag_ratio=1)
        # divide gray by morphology image

        if d != []:
            # print('deblured', d)  # (poner subrutina)
            text, conf = d2text(d)
            list_num.append(text)
            list_conf.append(conf)
        if d == [] or conf < 0.999:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            smooth = cv2.GaussianBlur(gray, (95, 95), 0)
            division = cv2.divide(gray, smooth, scale=255)
            squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            light = cv2.morphologyEx(division, cv2.MORPH_CLOSE, squareKern)
            light = cv2.threshold(division, 0, 255,
                                  cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            # self.debug_imshow("Light Regions", light, waitKey=True)
            # gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,
            #              dx=1, dy=0, ksize=-1)

            d = reader.readtext(light, contrast_ths=0.2, adjust_contrast=0.7, min_size=1,
                                mag_ratio=1)
            if d != []:
                # print('deblured', d)  # (poner subrutina)
                text, conf = d2text(d)
                list_num.append(text)
                list_conf.append(conf)

    if d == [] or conf < 0.9999:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        smooth = cv2.GaussianBlur(gray, (5, 5), 0)
        d = reader.readtext(smooth, contrast_ths=0.2,  adjust_contrast=0.7, min_size=1,
                            mag_ratio=1)
        # divide gray by morphology image
        if d != []:
            # print('deblured', d)  # (poner subrutina)
            text, conf = d2text(d)
            list_num.append(text)
            list_conf.append(conf)
        text = recognizer.recognize(image)

        if text:
             list_num.append(text)
             if list_conf:
                 list_conf.append(conf+0.01)
             else:
                 list_conf.append(0.5)
                 conf = 0.5

       # image2 = imutils.resize(image_or, width=300)
       # text = recognizer.recognize(image2)

       # if text:
       #     list_num.append(text)
       #     if list_conf:
       #         list_conf.append(conf + 0.01)
       #     else:
       #         list_conf.append(0.5)

    list_num = [item for i,item in enumerate(list_num) if list_conf[i] != 0]
    list_conf = [item for i, item in enumerate(list_conf) if list_conf[i] != 0]
    list_conf = [item for i, item in enumerate(list_conf) if list_num[i] != 0]
    list_num = [item for i,item in enumerate(list_num) if list_num[i] != 0]

    print(list_num)
    print(list_conf)

    if list_num != [] and vote == 'hard':
        if len(list_num) == 1 and list_conf[0]>=0.5:
           #print('tamaó lista igual 1', len(list_num))
            text = list_num[0]
            conf = list_conf[0]
        elif len(list_num) == 1 and list_conf[0]<0.5:
            text = '0'
            conf = 0
        else:
           # print('lista mayora 1',len(list_num))
            text = max(set(list_num), key=list_num.count)
            list_num_a = np.array(list_num)
            list_conf_a = np.array(list_conf)

            if len(set(list_num)) == len(list_num): # si cada predicción un número cogemos el máximo mejor
                   # text = list_num[list_conf == max(list_conf)]
                   text = list_num_a[list_conf_a == max(list_conf_a)][0]
                   conf = max(list_conf_a)

            elif np.array(text).size == 1:

                    conf = max(list_conf_a[list_num_a == text])

            else:
                    conf_list = [max(list_conf_a[list_num_a == t]) for t in text]
                    text = list_num_a[list_conf_a == max(conf_list)][0]
                    conf = max(conf_list)

    elif list_num != [] and vote == 'soft':
        """ aquí pesando por probabilidades"""
    else:
        text = '0'
        conf = 0

        # VOTING METHOD? CONTAR CUANTAS VECES ME SALE UNO DE LOS NÚMEROS??
    print(text)
    return text, conf




def recognize_numbers_new(crop):
    image_or = cv2.imread(crop)
    image = imutils.resize(image_or, width=200)
    #image = resize_image(image_or)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)
    light = cv2.threshold(light, 0, 255,
                          cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F,
    #              dx=1, dy=0, ksize=-1)

    roi = cv2.threshold(light, 0, 255,
                        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    return roi



def d2text(d):
    if len(d) == 1:
      #  print(d)
        #if d[0][-2].isdigit():
       # text = d[0][-2]
        #conf = d[0][-1]
        if d[0][-2].isdigit():
            text = d[0][-2]
            conf = d[0][-1]
        elif True in [char.isdigit() for char in d[0][-2]]:
            text = ''.join([i for i in d[0][-2] if i.isdigit()])
            conf = d[0][-1]
        else: #mejro con exception
            text = '0'
            conf = 0


    else:
        for item in d:
            if item[-2].isdigit():
                text = item[-2]
                conf = item[-1]
            elif True in [char.isdigit() for char in item[-2]]:
                 text = ''.join([i for i in item[-2] if i.isdigit()])
                 #text = item[-2]
                 conf = item[-1]
            else: # mejro con exception
                text = '0'
                conf = 0

    return text, conf


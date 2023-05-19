import cv2
import pandas as pd
import numpy as np
from ast import literal_eval
import exifread
from datetime import datetime
import tqdm
#import face_recognition
from itertools import compress
import os
import shutil
from deepface import DeepFace
import tensorflow as tf 

physical_devices = tf.config.list_physical_devices('GPU') 
#tf.config.experimental.set_memory_growth(physical_devices[0], True)

def GetDate(imgfile):
    """returs exit date"""
    with open(imgfile, 'rb') as fh:
        tags = exifread.process_file(fh, stop_tag="EXIF DateTimeOriginal")
        dateTaken = tags["EXIF DateTimeOriginal"]
        date = str(dateTaken)
        date_obj = datetime.strptime(date, '%Y:%m:%d %H:%M:%S')
        timestamp = datetime.timestamp(date_obj)
    return timestamp


def ReadFile(file, path_or):
    """Returns data frame from csv
    :file string with cvs file
    :path  string with original images location"""

    df1 = pd.read_csv(file)
    df1.bib_number = df1.bib_number.apply(literal_eval)
    df1.bib_conf = df1.bib_conf.apply(literal_eval)
    df1.dorsal_conf = df1.dorsal_conf.apply(literal_eval)

    ll = []; ll2 = [];  ll3 = []
    for l, l2, l3 in zip(df1.bib_number, df1.bib_conf, df1.dorsal_conf):
        if l:
            ll.append(l) ;  ll2.append(l2); ll3.append(l3)
        else:
            ll.append(['0']) ;  ll2.append([0]) ;   ll3.append([0])
    df1.bib_number = ll ;    df1.bib_conf = ll2;     df1.dorsal_conf = ll3

    time = []
    for im in df1.image:
        name = im.split('_add_')[0] + '_add_.jpg'
        time.append(GetDate(os.path.join(path_or, name)))
    df1["timestamp"] = time
    print('----------------------------------')
    print('df generated with timestamp added')
    print('                                  ')

    return df1

def faceEncode(images, path):
    """images is a list of images to encode, returns list of encodings"""
    enc = [];    lista_img = []
    for img in images:
        image = face_recognition.load_image_file(os.path.join(path, img + '.jpg'))
        encoding = face_recognition.face_encodings(image)
        if encoding:
            enc.append(encoding[0])
            lista_img.append(img)
    return lista_img, enc


def FindMatchFaceRec(encod1, listenc):
    """find macth in faces with face_recgonition"""
    results = face_recognition.compare_faces(listenc, encod1[0])
    return results


def FindMatch(path, file, lista, conf_list,distance):
    """compares face in file with faces detected in images in lista
    returns list with imatges that are match with DeepFace"""

    lst = [];
    dst = []
    for file2, conf in zip(lista, conf_list):

        if file.split('_add_')[0] != file2.split('_add_')[0]:
            if conf[0] < 0.80:
                results = DeepFace.verify(os.path.join(path, file + '.jpg'), os.path.join(path, file2 + '.jpg'), model_name='Facenet512',
                                          detector_backend='retinaface', enforce_detection=False)
                # results2 = DeepFace.verify(os.path.join(path,file),os.path.join(path,file2),model_name = models[8], detector_backend='retinaface',enforce_detection= False)
                if results["distance"] < distance:
                    lst.append(file2)
                    dst.append(results["distance"])

    return lst, dst

def add_element(dict, key, value):
    """adds element to dict"""
    if key not in dict:
        dict[key] = []
    dict[key].append(value)

def PersonToImage(df1s, faces='True'):
        """canvia con la otra que aquí he transformado bib_number en intgerers y no lists"""

        df1s.image = [img.split('_add_')[0]+'_add_' for img in df1s.image]
        # print(df1s)
        myDict = {}
        lst_lb, lst_cb , lst_cd , lst_img , lst_ind = ([] for i in range(5))
        for img in df1s.image.unique():
            lst_img.append(img)
            lb, cb , cd = ([] for i in range(3))
            bibs = df1s.bib_number[df1s.image == img]
            confs_bib = df1s.bib_conf[df1s.image == img]
            confs = df1s.dorsal_conf[df1s.image == img]
            ind = df1s.indicator[df1s.image == img]
            if all(item == 0 for item in ind):
                lst_ind.append(0)
            else:
                lst_ind.append(1)
            for b, c, d in zip(bibs, confs_bib, confs):
                if b[0] != '0':
                    print('Appending',b)
                    if isinstance(b,list):
                        lb = lb + b
                        cb = cb + c
                        cd = cd + d
                    else:
                        lb.append(b);cb.append(c);cd.append(d)

            lst_lb.append(lb)
            lst_cb.append(cb)
            lst_cd.append(cd)
        lst_lb2 = [i for i in lst_lb if i != 0]
        lst_cb2 = [c for i, c in zip(lst_lb, lst_cb) if i != 0]
        lst_cd2 = [c for i, c in zip(lst_lb, lst_cd) if i != 0]

        myDict['image'] = lst_img
        myDict['bib_number'] = lst_lb2
        myDict['bib_conf'] = lst_cb2
        myDict['dorsal_conf'] = lst_cd2
        myDict['indicator'] = lst_ind
        df = pd.DataFrame(myDict)

        return df

def RelatesFacesToImages(csvfile, path_crops, path_or, distance):
    """ adds dorsal to non confident detections if face recognition
    can be done
    csv file: file with OCR outputs
    path_person: path of person crops
    path_or: original path with images

    SE puede optimizar, no haría falta chequear los dorsales con mucha confianza
    Y ahora que hago explode habría que evitar repetir imagenes, hay imagenes con 2 caras pero puede confundir
    y ralentiza el código
        """



    df = ReadFile(csvfile, path_or)

    dict_dorsal = {}; dict_distance = {}

    # las de más de una cara debería quitarlas del df trabajar con un df sin ellas, para que no
    # las incluya en la lista.
    df2 = df[(df.bib_number.str.len() < 2)]  # ESTA MAL
    #print(df2)

    for img, dorsal, conf, t in zip(df2.image, df2.bib_number, df2.bib_conf, df2.timestamp):
        if conf[0] >= 0.90:
            tmin = max(t - 10, min(df2.timestamp))
            tmax = min(t + 10, max(df2.timestamp))
            # print(lista)
            # si esta no se detecta no hace falta hacer o otro
            time_cond = (df2.timestamp > tmin) & (df2.timestamp < tmax)
            check_cond = ((df2.bib_conf.apply(lambda x: x[0] < 0.8)) | (df2.bib_number.str.startswith('0')))


            lista = df2.image[time_cond & check_cond].tolist()
            conf_list = df2.bib_conf[time_cond & check_cond].tolist()
            dorsal_list = df2.bib_number[time_cond & check_cond].tolist()
            print(img,lista, conf_list, tmin,tmax)

            # print(list1,lista_img)
            list_f, dist_f = FindMatch(path_crops, img, lista, conf_list, distance)
            if list_f:
                print(f'{img} is matched with {list_f}')
                print(f'{img} distances {dist_f}')

                for im,dist in zip(list_f,dist_f):
                    add_element(dict_dorsal, im, dorsal[0])
                   # add_element(dict_distance, im, dist)
    # we have a dict_dorsal with dorsals associated to images, then we will take the most common
    df3 = df.copy()
    print(df3)
    # add pandas column of not changed
    df3["bib_number_exif"] = 'not changed'

    #for k, d, dist in zip(dict_dorsal.items(),dict_distance.values()):
    for k, d in dict_dorsal.items():

        # only dd if no identic to dorsal, para no añadir esos malos
        #dd = d[dist == max(dist)] probar esto
        dd = max(set(d), key=d.count)
        print('HOLA',k,dd)
        if df3.bib_conf[df3.image == k].values[0][0] < 0.9:
            print('changing all')
            df3.bib_number_exif[df3.image == k] = 'changed'
            df3.indicator[df3.image == k] = 1
            row_index = df.loc[df3.image == k].index[0]
           # Assign the new element as a list to the specified row and column
            df3.at[row_index, 'bib_number'] = [dd]
            df3.at[row_index, 'bib_conf'] = ['1.11']
            #df3.bib_number[df3.image == k] = dd
            df3.at[row_index, 'dorsal_conf'] = ['0.555']

           # df3.dorsal_conf[df3.image == k] = ['0.555']


    new = len(dict_dorsal.keys())


    print('---------------------------')
    print(f"End Face Matching, identified {new} new recognitions ")
    print('                           ')

#    df3.to_csv(csvfile.split('.')[0] + '_with_faces_person_faces_dist_015_all_proposals.csv')
    df_final = PersonToImage(df3)
    df_final.to_csv(csvfile.split('.')[0] + '_with_faces_dist_015_int10_test_check_cond2.csv')
    return df_final


def JoinFiles(path1, path2):
    """inpath should be file of results as in BlurrDetection.py """
    print("version deprecated valid only for raphael v1")

    pred_list, conf_list, bib_list ,ind_list = ([] for i in range(4))

    df1 = pd.read_csv(path1)
    df1.bib_number = df1.bib_number.apply(literal_eval)
    df1.bib_conf = df1.bib_conf.apply(literal_eval)
    df1.dorsal_conf = df1.dorsal_conf.apply(literal_eval)

    df2 = pd.read_csv(path2)
    df2.bib_number = df2.bib_number.apply(literal_eval)
    df2.bib_conf = df2.bib_conf.apply(literal_eval)
    df2.dorsal_conf = df2.dorsal_conf.apply(literal_eval)

    for im1, im2, pred1, pred2, conf_dor1, conf_dor2, conf_bib1, conf_bib2, ind1, ind2 in zip(df1.image, df2.image,
                                                                                              df1.bib_number,
                                                                                              df2.bib_number,
                                                                                              df1.dorsal_conf,
                                                                                              df2.dorsal_conf,
                                                                                              df1.bib_conf,
                                                                                              df2.bib_conf,
                                                                                              df1.indicator,
                                                                                              df2.indicator):
        pred, conf_bib, conf_dor = JoinPreds(pred1, pred2, conf_bib1, conf_bib2, conf_dor1, conf_dor2)
        # print(im1, im2, pred1,pred2,pred)

        if not pred:
            ind_list.append(0)
        elif pred == 0:
            ind_list.append(0)
        else:
            ind_list.append(1)

        pred_list.append(pred)
        conf_list.append(conf_dor)
        bib_list.append(conf_bib)
    # print(len(pred_list))
    df_new = pd.DataFrame().reindex_like(df1)
    df_new["image"] = df1["image"]
    df_new["bib_number"] = pred_list
    df_new["bib_conf"] = bib_list
    df_new["dorsal_conf"] = conf_list
    df_new["indicator"] = ind_list
    return df_new





def moveFile(im, lista, path, path_aux, write_text=False, text='0', color='red'):
    if not os.path.exists(path_aux):
        os.mkdir(path_aux)
    for l in lista:
        if im in l:
            source = os.path.join(path, l)
            destination = os.path.join(path_aux, l)

            # print('destination',destination)
            if write_text:
                AddTextCopy(source, destination, text, color)
            else:
                shutil.copy(source, destination)

def blurryImages(df, lista, path_or,path_out):
    path_aux = os.path.join(path_out, 'blurry')
    if not os.path.exists(path_aux):
        os.mkdir(path_aux)

    for img in lista:
        # hacer con filter function
        match = False
        for entry in df["image"]:
            if  img.split('_add_')[0]  in entry:
                match = True
                break
           # print(match)
        if match == False:
            print('Imagen no en la lista, guardando en blurry')
            source =os.path.join(path_or ,img)
            destination = os.path.join(path_aux,img)
            shutil.copy(source, destination)

def removeFile(im, lista, path):

    for l in lista:
        if im in l:
            source = path + l
            print('removin source', source)
            os.remove(source)

def DistributePhotos(path, df,  path_out, thres_bib=0.8, write_text=False):
    """distributes fotos dentro de path segun
    resultados del df. If dorsal exists geneartes a folder with name dorsal and puts images inside
    path_out path where the folder will be.
    de df by dorsal or not identified will be directed to not_dorsal"""
    print("version valid only with faces")
    print('generating csv with not confident')
    import csv
    if not os.path.exists(path_out):
        os.mkdir(path_out)
    f = open(os.path.join(path_out,'sin_clasificar.csv'), 'w')
    writer = csv.writer(f)

# write a row to the csv file
    lista = os.listdir(path)
    blurryImages(df,lista,path,path_out)
    for im, list_bib, conf in zip(df.image, df.bib_number, df.bib_conf):
        color = [];    text = []
        #print(im, list_bib, conf)
        if not list_bib:
            print('not dorsal',im, list_bib, conf)
            path_aux = path_out + '/SIN_CLASIFICAR/'
            moveFile(im, lista, path, path_aux)
            writer.writerow([im,list_bib,conf])
        else:
            cont = 0
            for bib, c in zip(list_bib, conf):
                #print('iteration',im, bib, c)
                if isinstance(c, str):
                    #bib = eval(bib)
                    c = eval(c)
                text.append(str(bib) + ',' + str(c))
                if c > thres_bib:
                    path_aux = path_out + '/' + str(bib) + '/'
                    color.append("black")
                    moveFile(im, lista, path, path_aux)
                    print('moving to', bib, 'folder')
                else:
                    color.append("red")
            if "red" in color:
                print('not confident', im, bib, c)
                if cont == 0:
                    writer.writerow([im,list_bib,conf])
                    cont = 1
            #    path_aux = path_out + '/not_confident/'
            #    moveFile(im, lista, path, path_aux, write_text = True, text = text, color = color)
                path_aux = path_out + '/SIN_CLASIFICAR/'
                moveFile(im, lista, path, path_aux, write_text = write_text, text = text, color= color)
    f.close()
    print('process completed')

def DistributePhotosAfter(path, df,  path_out, thres_bib=0.8, write_text=False):
    """Dsitrubites files in df in folders with bib neam if thres> thres_bib
    after first classification"""

    if not os.path.exists(path_out):
        os.mkdir(path_out)
    s_df = pd.read_csv(os.path.join(path_out, 'sin_clasificar.csv'), usecols=[0,1,2], names=['names','num','conf'],header=None)
    lista = os.listdir(path)
    # df.bib_number = df.bib_number.apply(literal_eval)
    # df.bib_conf = df.bib_conf.apply(literal_eval)
    for im, list_bib, conf in zip(df.image, df.bib_number, df.bib_conf):
        print(im)
        color = [];
        text = []
        # print(im, list_bib, conf)
        for bib, c in zip(eval(list_bib), eval(conf)):
            # print('iteration',im, bib, c)
            print(bib)
            if bib != 0:
                text.append(str(bib) + ',' + str(c))
                if c > thres_bib:
                    path_aux = path_out + '/' + str(bib) + '/'
                    color.append("black")

                    moveFile(im, lista, path, path_aux)
                else:
                    color.append("red")
        print(color)
        print(not "red" in color)
        if not "red" in color:
            removeFile(im, lista, path)
            s_df = s_df.drop(s_df[s_df.names==im].index)
    s_df.to_csv(os.path.join(path_out, 'sin_clasificar.csv'), index =False)

    print('process completed')

def AddTextAllImage(path, file, thres_bib=0.8, path_out='.', write_text=False):
    if not os.path.exists(path_out):
        os.mkdir(path_out)
    lista = os.listdir(path)
    df = pd.read_csv(file)
    df.bib_number = df.bib_number.apply(literal_eval)
    df.bib_conf = df.bib_conf.apply(literal_eval)
    for im, list_bib, conf in zip(df.image, df.bib_number, df.bib_conf):
        color = [];
        text = []
        # print(im, list_bib, conf)
        if not list_bib:
            path_aux = path_out + '/not_dorsal/'
            moveFile(im, lista, path, path_aux)
        for bib, c in zip(list_bib, conf):
            # print('iteration',im, bib, c)
            text.append(str(bib) + ',' + str(c))
            if c > thres_bib:
                path_aux = path_out + '/' + str(bib) + '/'
                color.append("black")
                # moveFile(im,lista,path,path_aux)
            else:
                color.append("red")
            path_aux = path_out + '/all_images_text/'
            moveFile(im, lista, path, path_aux, write_text=write_text, text=text, color=color)

def JoinPreds(pred1, pred2, conf1, conf2, cdor1, cdor2):
    # print(pred1,pred2,conf1,conf2,cdor1,cdor2)
    if pred1 != []:
        if pred1[0] != '':
            pred1 = [int(p) for p in pred1]
            pred1 = np.array(pred1)
    if pred2 != []:
        if pred2[0] != '':
            pred2 = [int(p) for p in pred2]
            pred2 = np.array(pred2)

    pred = [];
    conf = [];
    conf_dor = []
    conf2 = np.array(conf2);
    cdor2 = np.array(cdor2)

    for val, c, cd in zip(pred1, conf1, cdor1):

        if val in pred2:
            # print(pred2,conf2[pred2 == val])
            if len(conf2[pred2 == val]) > 1:
                c2 = max(conf2[pred2 == val])
                cd2 = max(cdor2[pred2 == val])

            else:
                c2 = float(conf2[pred2 == val])
                cd2 = float(cdor2[pred2 == val])
            if c2 < c:
                pred.append(val)
                conf.append(c)
                conf_dor.append(cd)
            else:
                pred.append(val)
                conf.append(c2)
                conf_dor.append(cd2)
        else:
            pred.append(val)
            conf.append(c)
            conf_dor.append(cd)

    for val, c2, cd2 in zip(pred2, conf2, cdor2):
        if val not in pred:
            pred.append(val)
            conf.append(c2)
            conf_dor.append(cd2)

    # print('loro',pred1,pred2,pred,conf,conf_dor)
    conf_dor = [c for i, c in zip(pred, conf_dor) if i != 0]
    conf = [c for i, c in zip(pred, conf) if i != 0]

    pred = [i for i in pred if i != 0]
    return pred, conf, conf_dor

def AddTextCopy(source:str,destination:str, text:list, color:list):
    """takes image and adds text in the top left corner"""
    from PIL import ImageFont, ImageDraw, Image

    image = cv2.imread(source)
#image = np.zeros((512,512,3), np.uint8)
# Convert to PIL Image
    cv2_im_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im_rgb)
    draw = ImageDraw.Draw(pil_im)

# Choose a font
#font = ImageFont.truetype("Roboto-Regular.ttf", 50)
    #font = ImageFont.truetype("Avenir.ttc", size=72)

# Draw the text
    y = 100
    for t,col in zip(text,color):
        #draw.text((100, y), str(t), font = font,fill=col)
        draw.text((100, y), str(t), fill=col)
        y += 100

# Save the image
    cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)
    cv2.imwrite(destination, cv2_im_processed)


import os
import numpy as np
from old_versions.ocr_functionsv5 import treatment

def read_dir(inpath):
    """reads directory and return list
    if directory does not exists or it is empty it
    returns an empty list"""
    l_im = []
    if os.path.exists(inpath):
        l_im = os.listdir(inpath)
        l_im.remove('crops')
        l_im.remove('labels')
    return l_im

def analyse_list(inpath, outpath, thres):
    """ Reads list with processed YOLO images
      and returns the outcomes
      0 : non detected dorsal
      1 : detected dorsal but not recognized numbers
      2 : detected dorsal and recognized numbers
      If 0 no other information is given
      If 1, list of confidence for each dorsal is returned
      If 2, list of confidences per dorsal, list of numbers and list of confidence"""
    import csv
    f = open(outpath, 'w')
    writer = csv.writer(f)
    writer.writerow(['image', 'indicator', 'bib_number', 'bib_conf', 'dorsal_conf'])
    l_im = read_dir(inpath)
    for image in l_im:
        # Opens a image in RGB mode
        basename = os.path.splitext(image)[0]
        labFile = inpath + '/labels/' + basename + '.txt'
        out = checkDorsal(labFile)
        Num = []; ConfNum =[]; ConfDor = []
        if out == 1:
            arr = np.loadtxt(labFile)
           # print('Basename', basename)
            CropList, ConfDor = MakeCropList(arr, basename, inpath,thres)
           # print('lista de crops',basename,CropList)
            Num, ConfNum, out = RecognizeBibNumber(CropList)
        writer.writerow([basename,out, Num, ConfNum, ConfDor])



def checkDorsal(LabFile):
      """If labFile does not exist, dorsal has not been detected and return 0"""
      out = 0
      if os.path.exists(LabFile):
          out = 1
      return out

def MakeCropList(listLab,basename, inpath,thres):
      """ from list of labels in one images make crop list with dorsals with confidence > thres"""
      conf_list = [];  crop_list = []
      #print('lista Labels', basename, listLab)
      if (listLab.ndim == 1):
         if listLab[-1] > thres:
             conf_list.append(listLab[-1])
             crops = [filename for filename in os.listdir(inpath + 'crops/dorsal/') if basename in filename]
             crop_list.append(inpath + 'crops/dorsal/' + crops[0])
      else:
         crops = [filename for filename in os.listdir(inpath + 'crops/dorsal/') if basename in filename]
         #print('lista Labels',basename,listLab)
         for i, conf in enumerate(listLab[:,-1]):
              if conf > thres:
                  conf_list.append(conf)
                  crop_list.append(inpath + 'crops/dorsal/' + crops[i])
      return crop_list,conf_list

def RecognizeBibNumber(crop_list):
     """Detect bib numbers using easyocr"""
     numbers = [];  number_confidence = []; out = 1
     for crop in crop_list:
    # preprocessing before apply easy_ocr
    # deblurGAN, DeepDeBlur
         print(crop)
         d = treatment(crop)
         print(d)
         if d == 1: # Javier
               numbers.append(0)
               number_confidence.append(0)
         else:
               d = d[0]
               numbers.append(d[0][1])
               number_confidence.append(d[0][2])
             #  print(d[0][1])
         #if reader.readtext(crop) and conf > thres:
         #     if len(reader.readtext(crop)) == 1:
         #         d = list(reader.readtext(crop,allowlist ='0123456789')[0])
         #     else:
         #        for i, text in enumerate(reader.readtext(crop)):
         #             if text[-2].isdigit():
         #                  d = text
         #             elif True in [char.isdigit() for char in text[-2]]:
         #                  d = list(reader.readtext(crop, allowlist='0123456789')[0])

     if not all([ v == 0 for v in numbers ]):
         out = 2
     print('lolo',numbers)
     return numbers, number_confidence, out

def read_outputs(predictions):
    """read_file"""
    import pandas as pd
    df = pd.read_csv('/Users/casaponsa/WORK/DDS_MODELS/SPIID/outputs3.csv')

    with open(predictions) as csvfile:
         spamreader = csv.reader(csvfile)
         type(spamreader)
         for row in spamreader:
              print(row)
        #reader.writerow([basename,out, Num, ConfNum, ConfDor])

def drawText(IMAGE_PATH, result):
    """to be adapted, result : ouput of easy ocr reader"""
    top_left = tuple(result[0][0][0])
    bottom_right = tuple(result[0][0][2])
    text = result[0][1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.imread(IMAGE_PATH)
    img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 3)
    img = cv2.putText(img, text, bottom_right, font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    return img


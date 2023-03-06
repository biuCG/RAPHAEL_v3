import numpy as np
import cv2
import easyocr
import math
from typing import Tuple, Union
from deskew import determine_skew
reader = easyocr.Reader(['es'])


#deskew function implementation, from notebook published in deskew library official 
#repository https://github.com/sbrunner/deskew, used in the following functions

def rotate(
        image: np.ndarray, angle: float, background: Union[int, Tuple[int, int, int]]
) -> np.ndarray:
    old_width, old_height = image.shape[:2]
    angle_radian = math.radians(angle)
    width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
    height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    rot_mat[1, 2] += (width - old_width) / 2
    rot_mat[0, 2] += (height - old_height) / 2
    return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

def pred(im,a,b,c,d):
    
    #easyocr.readtext method modification to use by the treatment function:
    
    #If one single reading, we force reading integers, otherwise we turn to integer the part we can, 
    #as a prediction,and return the rest as additional possibilities to study.
    
    r1=reader.readtext(im,contrast_ths =a,adjust_contrast=b,min_size=c,mag_ratio=d)
    
    r=0
    rd1=[]
    rd2=[]
    if len(r1)>1:
        for iel,el in enumerate(r1):
            try:
                r=int(l[1])
                rd1.append(el)
            except:
                rd2.append(el)
    else:
        rd1=reader.readtext(im,allowlist="0123456789",contrast_ths =a,adjust_contrast=b,min_size=c,mag_ratio=d)
        
    return [rd1,rd2]
    

def treatment(imgi,endpth = None):

    or_im = cv2.imread(imgi)
    color_1=pred(or_im,0.2,0.7,1,1)
    color_2=pred(or_im,0.2,0.7,15,15)
    color=[color_1[0]+color_2[0],color_1[1]+color_2[1]]
#     print(color)

    #Reads the image to a file and runs OCR on image without any treatment.
    #If no detection, applies grayscale.
    
    if any([len(color[0])==0,color==None]):
        gr = cv2.cvtColor(or_im, cv2.COLOR_BGR2GRAY)
        gray_1=pred(gr,0.2,0.7,1,1)
        gray_2=pred(gr,0.2,0.7,15,15)
        gray=[gray_1[0]+gray_2[0],gray_1[1]+gray_2[1]]
#         print(gray)
        
        #Reads the image to a file and runs OCR on image without any treatment.
        #If no detection, applies deskew.
        
        if any([len(gray[0])==0,gray==None]):
            angle = determine_skew(or_im)
            rotated = rotate(gr, angle if angle!=None else 0, (0, 0, 0))
            graydesk_1=pred(rotated,0.2,0.7,1,1)
            graydesk_2=pred(rotated,0.2,0.7,15,15)
            graydesk=[graydesk_1[0]+graydesk_2[0],graydesk_1[1]+graydesk_2[1]]
#             print(graydesk)
                
            #Reads the image to a file and runs OCR on image without any treatment.
            #If no detection, treats gradients.
            
            if any([len(graydesk[0])==0,graydesk==None]):
                vf = np.abs(255-rotated)
                gradientfree=cv2.Laplacian(vf,cv2.CV_8UC3)
                graydesklapl_1=pred(gradientfree,0.2,0.7,1,1)
                graydesklapl_2=pred(gradientfree,0.2,0.7,15,15)
                graydesklapl=[graydesklapl_1[0]+graydesklapl_2[0],graydesklapl_1[1]+graydesklapl_2[1]]
#                 print(graydesklapl)
                    
                #Reads the image to a file and runs OCR on image without any treatment.
                #If no detection, returns 1.
                
                if any([len(graydesklapl[0])==0,graydesklapl==None]):
                    return 1
                
                else:
                    if endpth:
                         cv2.imwrite(endpth+"/"+"graydesklapl_"+im, gradientfree)
                    graydesklapl.append("graydesklapl")
#                     print(graydesklapl)
                    return graydesklapl
            else:
                if endpth:
                    cv2.imwrite(endpth+"/"+"graydesk_"+im, rotated)
                graydesk.append("graydesk")
#                 print(graydesk)
                return graydesk
        else:
            if endpth:
                cv2.imwrite(endpth+"/"+"gray_"+im, gr)
            gray.append("gray")
#             print(gray)
            return gray
    else:
        if endpth:
             cv2.imwrite(endpth+"/"+"color_"+im, or_im)
        color.append("color")
#         print(color)
        return color
                

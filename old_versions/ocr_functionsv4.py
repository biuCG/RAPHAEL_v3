import numpy as np
import pandas as pd
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

def checkaddlist(charrecognised):
    try:
        test=int(charrecognised)
        r=charrecognised
    except:
        r= None
    return r

def pred(im,thmin_tries,thmin_recon):
    #thmin_tries = [threshold1, threshold2] If confidence threshold1 is not surpassed results
    #tries again with min_size, mag_ratio values to 15, if threshold2 is still not surpassed 
    #after this, text is detected as string and every character different from an integer removed
    #manually
    
    r1=pd.DataFrame(reader.readtext(im,contrast_ths =0.2,allowlist="0123456789",adjust_contrast=0.7,min_size=1,mag_ratio=1),columns=["window","text","confidence"])
    
    m1=r1["confidence"].max()
    if m1<thmin_tries[0]:
        r1=r1.append(pd.DataFrame(reader.readtext(im,allowlist="0123456789",contrast_ths =0.2,adjust_contrast=0.7,min_size=15,mag_ratio=15),columns=["window","text","confidence"]),ignore_index=True)
    
    m1=r1["confidence"].max()
    if m1<thmin_tries[1]:
        r2=pd.DataFrame(reader.readtext(im,contrast_ths =0.2,adjust_contrast=0.7,min_size=1,mag_ratio=1),columns=["window","text","confidence"])
        r2=r2.append(pd.DataFrame(reader.readtext(im,contrast_ths =0.2,adjust_contrast=0.7,min_size=15,mag_ratio=15),columns=["window","text","confidence"]),ignore_index=True)
        for iel,el in enumerate(r2["text"]):
            if checkaddlist(el.strip())==None:
                rn=[checkaddlist(c) for c in el.strip() if checkaddlist(c)!=None]
                r=''.join(rn) if ''.join(rn)!='' else None
                if ''.join(rn)=='':
                    r2=r2.loc[r2["text"]!=el,:]
                else:
                    r2.loc[iel,"text"]=''.join(rn)
        r1=r1.append(r2,ignore_index=True)
        
    m2=r1["confidence"].max()
    
    if m2>=thmin_recon:
        r1["predlength"]=r1.loc[:,"text"].agg(len)
        if checkaddlist(r1.loc[(r1["confidence"]==r1.loc[r1["predlength"]>1,"confidence"].max()),"text"].values)!=None:
            r1["thaccept"]=np.where(r1["confidence"]==r1.loc[r1["predlength"]>1,"confidence"].max() if r1.loc[r1["predlength"]>1,:].shape[0]>1 else r1["confidence"]==m2, True, False)
        else:
            r1["thaccept"]=np.repeat(False,r1.shape[0])
        r1=r1.drop(['predlength'],axis=1)
    else:
        r1["thaccept"]=np.repeat(False,r1.shape[0])
        
    rd1=r1.loc[r1["thaccept"]==True,:]
    rd2=r1.loc[r1["thaccept"]==False,:]
    rd1=list(rd1.drop(['thaccept'],axis=1).to_records(index=False))
    rd2=list(rd2.drop(['thaccept'],axis=1).to_records(index=False))
    
    return [rd1,rd2]


def treatment(im,thmin_tries=[0.9,0.9],thmin_recon=0):

    #Output will be a dataframe with the following fields:
    #window 	text 	confidence 	 treatment
    
    or_im=cv2.imread(im)
    color=pred(or_im,thmin_tries,thmin_recon)
        
    if any([len(color[0])==0,color==None]):
        gr = cv2.cvtColor(or_im, cv2.COLOR_BGR2GRAY)
        gray=pred(gr,thmin_tries,thmin_recon)
        
        if any([len(gray[0])==0,gray==None]):
            angle = determine_skew(or_im)
            rotated = rotate(gr, angle if angle!=None else 0, (0, 0, 0))
            graydesk=pred(rotated,thmin_tries,thmin_recon)
            
            if any([len(graydesk[0])==0,graydesk==None]):
                vf = np.abs(255-rotated)
                gradientfree=cv2.Laplacian(vf,cv2.CV_8UC3)
                graydesklapl=pred(gradientfree,thmin_tries,thmin_recon)
                
                if any([len(graydesklapl[0])==0,graydesklapl==None]):
                    return 1
                
                else:
                    graydesklapl.append("graydesklapl")
                    return graydesklapl
            else:
                graydesk.append("graydesk")
                return graydesk
        else:
            gray.append("gray")
            return gray
    else:
        color.append("color")
        return color
                

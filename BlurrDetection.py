import os
from blur_detector import detect_blur_fft, ValidImage
import imutils
import cv2
import tqdm
import shutil


def main(src_path: str, out_path: str, threshold: float = 10):
    """scr path: main path where object crops are inside this path
       we will find crops and labels
      """
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
   # if not os.path.isdir("blurry_detections/blur_images"):
    #    os.mkdir("blurry_detections/blur_images")
    src_crops = os.path.join(src_path, 'crops/person')
    src_labels = os.path.join(src_path, 'labels')

    for label in tqdm.tqdm(os.listdir(src_labels)):
         im_list = ValidImage(src_labels, label)
         print('label',im_list)
         if im_list:
            for file in im_list:
                 print(out_path,file)
                 shutil.move(os.path.join(src_crops,file), os.path.join(out_path, file))
    print("Person crops small or narrow deleted, checking for blurry images")
    # with the rest of the images w
    filenames = [os.path.join(src_crops, f ) for f in os.listdir(src_crops)]
    for file in tqdm.tqdm(filenames):
            print(file,label)
            img = cv2.imread(file)
            el = imutils.resize(img, width=500)
            gray = cv2.cvtColor(el, cv2.COLOR_BGR2GRAY)
            (mean, blurry) = detect_blur_fft(gray, size=60,
			                    thresh=threshold)

            if blurry:
              #  tqdm.tqdm.write(f"La imagen {file} contiene, al menos, una persona no borrosa.")
                shutil.move(file, os.path.join(out_path, os.path.split(file)[1]))



    #os.rename(src_path, src_path + '_blurry')
    #os.rename(out_path, src_path)



# Press the green button in the gutter to run the script.
import sys
if __name__ == '__main__':
        main(src_path=sys.argv[1], out_path=sys.argv[2])

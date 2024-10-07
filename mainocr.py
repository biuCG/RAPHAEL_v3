import ocr
import os
full_path = 'data_ocr'
#ocr.recognize_numbers(full_path)
import pytesseract
import pandas as pd
from PIL import Image

#for root, dirs, files in os.walk(full_path):
#      for i in files:
#         print('data_ocr/'+ i)
#         my_image = ocr.recognize_numbers_new('data_ocr/'+ i)

#          custom_config = r'--oem 3 --psm 6 outputbase digits'
#          text = pytesseract.image_to_string(my_image, config=custom_config)
#          print(text)
          # Get the TSV output from Tesseract
          #image = Image.open('data_ocr/'+ i)
          #image_ch= ocr.resize_image(image)
          #tsv_output = pytesseract.image_to_string(image_ch, config=custom_config)
# Display the dataframe containing the recognized text, confidence scores, etc.
          #print(tsv_output)

#my_list = []

for root, dirs, files in os.walk(full_path):
      for i in files:
        print('data_ocr/'+ i)
        ocr.recognize_numbers('data_ocr/'+i)

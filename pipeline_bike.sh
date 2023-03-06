# Local file Biuse
# git clone https://github.com/ultralytics/yolov5  # clone
#pip install -qr yolov5/requirements.txt

# changes with raphael 1
# * performs dorsal recognition in persons crops not images
# * includes selection of small and narrow person crops (normalized area <0.003 and w/h < 0.14 are dropped)
# * match faces in intervals of +/-20 seconds using exif information

# valid for bikes and runners
# Bikes weights/*bike.pt img size 640  need to decide weights yolov5x or yolov5s6
# For bikes distance in BibDistributionInFoldersAddingFaces.py max 0.1 for runners distance max 0.2
# images should be JPG!! launch error instead


root=~/RAPHAEL/BIKES/
yolo_path=~/RAPHAEL/yolov5/
model=yolov5s6
weights=~/RAPHAEL/yolo_weights/
test_name=DEMOBICI_validation_dataset
out_path=$root/${test_name}_YOLO_person_detection/
out_name1=${model}

image_path=${root}/${test_name}

# Falta incluir lo de Carlos de añadir _ delante


# people cropping
python3 ${yolo_path}/detect.py --weights ${weights}/yolov5s.pt --img 640 --source $image_path --save-conf --save-crop --save-txt  --project $out_path --name res

# para people_croppin2 he quitado lowe resolution people y cut
res_path=${out_path}/res

# Blurr Detections
# en realidad carpeta de salida ya no hace falta, solo para ver de momento
python BlurrDetection.py $res_path  ${res_path}'/crops/person_blurry'

# image path con personas solo y sin borrosas
crop_path=${res_path}/crops/person
out_path=${root}/${test_name}_YOLO_dorsal/


python3 ${yolo_path}/detect.py --weights ${weights}/${model}_bike.pt --img 1280  --conf 0.2 --source $crop_path --save-conf --save-crop --save-txt --project $out_path --name $out_name1

#pip install -qr requirements.txt

path1=${out_path}/${out_name1}/
res_file1=${out_path}/results_${out_name1}.csv

python3 main.py --inpath $path1 --output_file $res_file1 --thres 0.2


out_path_final=${out_path}/folders_with_faces_${model}/
image_path=${root}/${test_name}

python3 BibDistributionInFoldersAddingFaces.py --csv1 $res_file1 --path_crops ${crop_path} --input_path ${image_path} --output_path $out_path_final --thres 0.8 --write_text True




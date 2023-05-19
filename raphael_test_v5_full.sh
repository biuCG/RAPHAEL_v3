#!/bin/bash -x 

#Variables
input="CHECK_FACES_RUN"

app_path=/home/biuse/RAPHAEL/RAPHAEL-master_v4
working_directory=${HOME}/RAPHAEL/TEST_V4_FULL_CHECK_FACES/

echo "Creando directorio de trabajo... "
echo ${working_directory}
mkdir -p ${working_directory}
cd ${working_directory}
mkdir images
cp /home/biuse/RAPHAEL/${input}/* images/
[ $? -eq 0 ]  || exit 1

echo "Descargando ficheros ..."
# Set space as the delimiter
IFS=','

cd images

#Rename files
count=`ls -1 *.jpg 2>/dev/null | wc -l`
if [ $count != 0 ]; then for file in *.jpg; do mv -- "$file" "${file%.jpg}_add_.jpg"; done; fi
count=`ls -1 *.JPG 2>/dev/null | wc -l`
if [ $count != 0 ]; then for file in *.JPG; do mv -- "$file" "${file%.JPG}_add_.JPG"; done; fi

cd ..
################### Ejecucion ###################################################################
export PYTHONPATH="${PYTHONPATH}:/home/biuse/RAPHAEL/RAPHAEL-master_v4"

model='best_crops'
out_path=${PWD}/output/
yolo_path=${app_path}/yolov5
weights=${app_path}/weights/
image_path=${PWD}/images/

out_name1=${model}

file=$(ls ${image_path} | head -1)
val=$(identify -format '%w ' ${image_path}/$file)

python3 ${yolo_path}/detect.py --weights ${weights}/yolov5s.pt --img 640 --source $image_path --save-conf --save-crop --save-txt  --project $out_path --name res

res_path=${out_path}/res

# Blurr Detections
# en realidad carpeta de salida ya no hace falta, solo para ver de momento
python ${app_path}/BlurrDetection.py $res_path  ${res_path}'/crops/person_blurry'

# image path con personas solo y sin borrosas
crop_path=${res_path}/crops/person
out_path2=${out_path}/YOLO_dorsal/


python3 ${yolo_path}/detect.py --weights ${weights}/${model}_run.pt --img 1280  --conf 0.2 --source $crop_path --save-conf --save-crop --save-txt --project $out_path2 --name $out_name1
[ $? -eq 0 ]  || exit 1

#pip install -qr requirements.txt

path1=${out_path2}/${out_name1}/
res_file1=${out_path}/results_${out_name1}.csv

python3 ${app_path}/main.py --inpath $path1 --output_file $res_file1 --thres 0.2
[ $? -eq 0 ]  || exit 1


out_path_final=${out_path}/folders_with_faces_${model}/

python3 ${app_path}/BibDistributionInFoldersAddingFaces.py --csv1 $res_file1 --path_crops ${crop_path} --input_path ${image_path} --output_path $out_path_final --thres 0.8 --write_text False --th_faces 0.3


model=yolov5x
image_path=${out_path_final}/SIN_CLASIFICAR/
out_path=${out_path}
out_name=${model}
echo ${app_path}
python3 ${yolo_path}/detect.py --weights $weights/raphael_${model}.pt --img 640 --conf 0.2 --source $image_path --save-conf --save-crop --project ${out_path} --name ${out_name}
 
path1=${out_path}/${out_name}/
res_file=${out_path}/results_${out_name}.csv

cd app_path
python3 ${app_path}/main.py --inpath $path1 --output_file $res_file --thres 0.2


python3 ${app_path}/BibDistributionInFoldersAddingFaces.py --csv1 $res_file1 --path_crops ${crop_path} --input_path ${image_path} --output_path $out_path_final --thres 0.8 --write_text False 



##################################################################################################

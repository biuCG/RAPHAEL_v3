git clone https://github.com/ultralytics/yolov5  # clone
pip install -qr yolov5/requirements.txt

model='yolovs6l'
test_name='data'
out_path='test'/${test_name}/
out_path_final=${out_path}/folders/  # filw where folders with pictures will b

weights=weights/weights_${model}.pt
image_path=${test_name}/
e

out_name1=${model}
out_name2=${model}_augmented

file=$(ls ${image_path} | head -1)
val=$(identify -format '%w ' ${image_path}/$file)

# Dorsal Detection

python3 yolov5/detect.py --weights $weights --img 1280 --conf 0.1 --source $image_path --save-conf --save-crop --save-txt --project $out_path --name $out_name1

python3 yolov5/detect.py --weights $weights --img $val --conf 0.1 --source $image_path --augment --save-conf --save-crop --save-txt --project $out_path --name $out_name2


# Bib recognition
pip install -qr requirements.txt

path1=${out_path}/${out_name1}/
path2=${out_path}/${out_name2}/
res_file1=${out_path}/results_${out_name1}.csv
res_file2=${out_path}/results_${out_name2}.csv

python3 main.py --inpath $path --output_file $res_file1 --thres 0.1
python3 main.py --inpath $path --output_file $res_file2 --thres 0.1

# Join results and distribute pictures
python3 BibDistributionInFolders.py --csv1 $res_file1 --csv2 $res_file2 --input_path $image_path --output_path $out_path_final --thres 0.8 --write_text False


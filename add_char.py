import os
import shutil 


#shutil.copytree('/mnt/s3/racephotos/DEMOBICI_validation_dataset',path)
path ='~/RAPHAEL/BIKES/DEMOBICI_validation_dataset'
l1_im = os.listdir(path)
print(l1_im)
for image in l1_im:
    print(image)
    basename = os.path.splitext(image)
    old_name = os.path.join(path, image)
    new_name = os.path.join(path, basename[0] +'_'+basename[1])
    #print(new_name)
    #os.rename(old_name, new_name)

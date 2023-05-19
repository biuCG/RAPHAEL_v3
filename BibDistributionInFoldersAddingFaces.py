import utilsDist as utils
import argparse
import tqdm
import pandas as pd

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv1', default = 'test1.csv', type=str, help='csv file with bib at YOLO training resolution')
    parser.add_argument('--path_crops', default = '.', type=str, help='file with crops')
    parser.add_argument('--input_path', type=str, default='data', help='path with original images')
    parser.add_argument('--output_path', type=str, default='.', help='path of folders generated ')
    parser.add_argument('--thres', type=float, default=0.8, help='confidence threshold of dorsals to be analysed')
    parser.add_argument('--write_text', type=bool, default=False, help='If True bib numbers are added to non-confident images')
    parser.add_argument('--th_faces', type=float, default = None, help='maximum distance between faces, null if not faces')

    opt = parser.parse_args()
    return opt

def GenerateFolders(csv1,path_crops,input_path,output_path,th_faces,thres = 0.8,write_text = False):
    """ Combines both csv files gerenated from 2 YOLO crop detection and then
    from that combination distributes the images in folders given by bib number"""

    if th_faces:
        df = utils.RelatesFacesToImages(csv1,path_crops,input_path, distance = th_faces)
    #print(input_path,output_path,thres,write_text)

        utils.DistributePhotos(input_path, df, output_path, thres, write_text)
    
    else:
        df=pd.read_csv(csv1)
        utils.DistributePhotosAfter(input_path, df, output_path, thres, write_text)

if __name__ == '__main__':
    opt = parse_opt()
   # print(vars(opt))
    GenerateFolders(**vars(opt))


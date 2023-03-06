# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import argparse

def detect_bibnumber(inpath, output_file, thres):
    """ Detect bibnumbers form images in directory and returns a written csv file
    :param inpath:  input path with dorsal detected with DL model
    :param output_file: output csv file with results: for each line image name, indicator, list of results:
      indicator:
      0 : non detected dorsal
      1 : detected dorsal but not recognized numbers
      2 : detected dorsal and recognized numbers
      results:
      If 0 no other information is given
      If 1, list of confidence for each dorsal is returned
      If 2, list of confidences per dorsal, list of numbers and list of confidenc
    :param thres:   threshold of confidence of dorsal detected minimum to try to recognize numbers
    """
    from utils import read_dir, analyse_list
    print(inpath)
    if read_dir(inpath):
        analyse_list(inpath, output_file, thres)
    else:
         print('Empty directory')


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', default = 'run/detect/exp1', type=str, help='Yolo output detect.py path')
    parser.add_argument('--output_file', type=str, default='output.csv', help='output csv file')
    parser.add_argument('--thres', type=float, default=0.1, help='confidence threshold of dorsals to be analysed')

    opt = parser.parse_args()
    return opt
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    opt = parse_opt()
    detect_bibnumber(**vars(opt))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

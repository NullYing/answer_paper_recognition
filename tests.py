import os
import sys

cur_path = os.path.abspath(__file__)
parent = os.path.dirname
sys.path.append(parent(parent(cur_path)))

from server.image_processing.scan import AnswerScan


if __name__ == '__main__':
    for dir_path, sub_paths, files in os.walk('img'):
        for f in files:
            file_path = os.path.join(dir_path, f)
            answer_scan = AnswerScan(file_path, debug=True)
            print(answer_scan.scan())
            break

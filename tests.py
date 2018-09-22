import os
import sys
from core.scan import AnswerScan


if __name__ == '__main__':
    for dir_path, sub_paths, files in os.walk('img'):
        for f in files:
            file_path = os.path.join(dir_path, f)
            answer_scan = AnswerScan(file_path, debug=True)
            print(answer_scan.scan())
            break

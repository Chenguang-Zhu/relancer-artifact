import collections
import csv
from macros import R_REPAIR_ACTION_MODEL_PREDICTION_CSV

def extractInfoLineFromErrMsg(err_msg):
    for i, l in enumerate(err_msg.split('\n')):
        if 'Error: ' in l and ' Error: ' not in l:
            info_line = l.strip()
            if info_line.endswith(':'):
                for j in range(i+1, len(err_msg.split('\n'))):
                    info_line += ' ' + err_msg.split('\n')[j]
            if not isErrorInfoLineInCSV(info_line):
                info_line = info_line.split(": ")[0] + ": PLACEHOLDER"
            return info_line

def isErrorInfoLineInCSV(info_line):
    fp = open(R_REPAIR_ACTION_MODEL_PREDICTION_CSV, 'r')
    csv_reader = csv.reader(fp, delimiter=",", quotechar='"')
    for row in csv_reader:
        error_info = row[0].strip()
        if error_info == info_line:
            return True
    return False

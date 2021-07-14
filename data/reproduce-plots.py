#!/usr/bin/python3

import sys
import os
import collections
import json
import subprocess as sub
import statistics
import random
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__)) # Dir of this script
LOGS_DIR = SCRIPT_DIR + '/exec-logs/'
SUBJECTS = SCRIPT_DIR + '/notebooks.txt'
EACH_TOOL_FIXING_TIME_JSON = SCRIPT_DIR + '/each-tool-run-time.json'
COLOR_MAP = "tab10"

TOOLS = ['relancer', 'github', 'apidoc', 'text', 'random', 'naive']

def getSubjects():
    subjects = []
    with open(SUBJECTS, 'r') as fr:
        lines = fr.readlines()
    for i in range(len(lines)):
        subject = lines[i].strip()
        if subject not in subjects:
            subjects.append(subject)
    return subjects

def extractExecTimeFromRepairLog(log_file):
    with open(log_file, 'r') as fr:
        lines = fr.readlines()
    repair_time = 'TO'
    if lines[-1].strip().startswith('[REPAIR EXEC TIME]: '):
        repair_time = float(lines[-1].strip().split()[-1])
        repair_time = round(repair_time, 2)
    return repair_time

def genEachToolFixingTimeJSON(subjects):
    each_tool_fixing_time_dict = collections.OrderedDict({})
    if os.path.isfile(EACH_TOOL_FIXING_TIME_JSON):
        with open(EACH_TOOL_FIXING_TIME_JSON, 'r') as fr:
            each_tool_fixing_time_dict = \
                json.load(fr, object_pairs_hook=collections.OrderedDict)
    for case in subjects:
        #print('--- Processing ' + case)
        each_tool_fixing_time_dict[case] = collections.OrderedDict({})
        project = case.split('/')[0]
        notebook = case.split('/')[1]
        for tool in TOOLS:
            repair_log_file = LOGS_DIR + '/' + tool + '/' + project + '/' + \
                notebook + '.repair.log'
            exec_log_file = LOGS_DIR + '/' + tool + '/' + project + '/' + \
                notebook + '.exec.log'
            if not os.path.isfile(repair_log_file):
                each_tool_fixing_time_dict[case][tool] = 'TO'
                each_tool_fixing_time_dict[case][tool + '_time'] = 'TO'
                continue
            repair_time = extractRepairTimeFromLog(repair_log_file)
            repair_result = extractRepairResultFromLog(exec_log_file)
            each_tool_fixing_time_dict[case][tool] = ''
            each_tool_fixing_time_dict[case][tool + '_time'] = repair_time
            if repair_result == False:
                each_tool_fixing_time_dict[case][tool + '_time'] = 'FAIL'
    with open(EACH_TOOL_FIXING_TIME_JSON, 'w') as fw:
        json.dump(each_tool_fixing_time_dict, fw, indent=2)
    return each_tool_fixing_time_dict

def extractRepairTimeFromLog(log_file):
    with open(log_file, 'r') as fr:
        lines = fr.readlines()
    if '[REPAIR EXEC TIME]: ' not in lines[-1]:
        return 'TO'
    repair_time = round(float(lines[-1].strip().split()[-1]), 2)
    return repair_time

def extractRepairResultFromLog(log_file):
    repair_result = True
    with open(log_file, 'r') as fr:
        lines = fr.readlines()
    if not lines:
        return repair_result
    if 'Error: ' in lines[-1] and ' Error: ' not in lines[-1]:
        repair_result = False
    return repair_result

def plotComponentsBarChart(each_tool_fixing_time_dict, color_map=COLOR_MAP):
    num_relancer_fixes = len([n for n in each_tool_fixing_time_dict \
                          if each_tool_fixing_time_dict[n]['relancer_time'] \
                              not in ['TO', 'FAIL']])
    num_github_fixes = len([n for n in each_tool_fixing_time_dict \
                          if each_tool_fixing_time_dict[n]['github_time'] \
                            not in ['TO', 'FAIL']])
    num_apidoc_fixes = len([n for n in each_tool_fixing_time_dict \
                          if each_tool_fixing_time_dict[n]['apidoc_time'] \
                            not in ['TO', 'FAIL']])
    sns.set()
    data = [num_relancer_fixes, num_github_fixes, num_apidoc_fixes]
    tools = ['RELANCER', r'RELANCER$_{github}$', r'RELANCER$_{doc}$']
    df = pd.DataFrame({'num': data},
                      index=tools)
    cmap = plt.get_cmap(color_map)
    colors = cmap(range(3))
    fig = plt.figure(figsize=(4.5, 4))
    ax = fig.add_subplot(111)
    df.plot(kind='bar', ax=ax, width=0.3, align='center', rot=0,
            legend=False)
    for p in ax.patches:
        ax.annotate(str(p.get_height()), (p.get_x() * 1.03, p.get_height() * 1.005))    
    ax.set_ylabel('# Fixed Notebooks')
    plt.subplots_adjust(bottom=0.1, left=0.15, top=1.0)
    fig.savefig('fig5b.eps')
    print ('\n----------------------------')    
    print ('1. Figure 5b data:\n')
    for t, d in zip(tools, data):
        print (t + ": " +  str(d))
    print ('----------------------------')
    #plt.show()    
    plt.clf()

TIME_RANGES = [1, 5, 10, 15, 20, 25, 30] # min
TIME_RANGES = [str(t) for t in TIME_RANGES]
COLOR_MAP = "Dark2"
def plotNumOfFixedNotebooksOverTime(notebook_fix_time_dict,
                                    tools=['relancer', 'random', 'text', 'naive'],
                                    time_ranges=TIME_RANGES, color_map=COLOR_MAP):
    tool_names = [r'RELANCER', r'RELANCER$_{random}$',
                  r'RELANCER$_{text}$', r'RELANCER$_{naive}$']    
    fix_time_data = collections.OrderedDict({})
    for tool in tools:
        fix_time_data[tool] = collections.OrderedDict({})
        for t in time_ranges:
            fix_time_data[tool][t] = 0
    for tool in tools:            
        for case in notebook_fix_time_dict:
            exec_time = notebook_fix_time_dict[case][tool + '_time']
            if exec_time == 'TO':
                continue
            if exec_time == 'FAIL':
                continue            
            exec_time_in_min = float(exec_time) / 60
            for t in time_ranges:
                if exec_time_in_min <= int(t):
                    fix_time_data[tool][t] += 1
    sns.set()
    df = pd.DataFrame(fix_time_data, columns=tools, index=time_ranges)
    df.columns = tool_names
    df.index.names = ['Time (min)']
    print ('----------------------------')    
    print ('2. Figure 6 data:\n')    
    print (df)
    print ('----------------------------')    
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    cmap = plt.get_cmap(color_map)
    colors = cmap(range(len(tools)))    
    df.plot(kind='line', ax=ax, colormap=color_map,
            style=['o-','.--','s:', '+-'])
    ax.legend(tool_names)
    ax.set(ylabel='#Fixed Notebooks')
    ax.set(xlabel='Execution Time (min)')
    fig.tight_layout()
    fig.savefig('fig6.eps')
    #plt.show()
    plt.clf() # MUST CLEAN    

if __name__ == '__main__':
    subjects = getSubjects()
    each_tool_fixing_time_dict = genEachToolFixingTimeJSON(subjects)
    plotComponentsBarChart(each_tool_fixing_time_dict)
    plotNumOfFixedNotebooksOverTime(each_tool_fixing_time_dict)

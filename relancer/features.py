import json
import math
import collections
import statistics

from macros import API_FQN_MAPPING_JSON_FILE
from macros import MAPPING_JSON_FILE
from macros import FEATURES_CSV_FILE
from macros import MANUAL_TRAINING_MAPPING_JSON_FILE

from macros import ERROR_MSG_DATA_JSON
from macros import RESULTS_DIR
from macros import OUR_RESULTS_DIR
from macros import SUBJECTS_FILE

def extractInfoFromMapping(mapping_file=API_FQN_MAPPING_JSON_FILE,
                           ground_truth_mapping_file=MAPPING_JSON_FILE,
                           mutation_ground_truth_file=MANUAL_TRAINING_MAPPING_JSON_FILE):
    with open(mapping_file, 'r') as fr:
        mapping = json.load(fr)
    with open(ground_truth_mapping_file, 'r') as fr:
        ground_truth_mapping = json.load(fr)
    with open(mutation_ground_truth_file, 'r') as fr:
        mutation_ground_truth_mapping = json.load(fr)
    for old_api in mapping:
        candidates = mapping[old_api]
        is_true_assigned = False
        if old_api in ground_truth_mapping:
            for cand in candidates:
                if cand in ground_truth_mapping[old_api] and not is_true_assigned:
                    candidates[cand]['correctness'] = True
                    is_true_assigned = True
                else:
                    candidates[cand]['correctness'] = False
        elif old_api in mutation_ground_truth_mapping:
            for cand in candidates:
                if cand in mutation_ground_truth_mapping[old_api] and not is_true_assigned:
                    candidates[cand]['correctness'] = True
                    is_true_assigned = True
                else:
                    candidates[cand]['correctness'] = False
    return mapping

def generateFeaturesCSV(features_csv_file=FEATURES_CSV_FILE):
    mapping_info = extractInfoFromMapping()
    with open(features_csv_file, 'w') as fw:
        lines = 'old_api_fqn,candidate_fqn,occurrence,occurrence_percentage,edit_distance_similarity,edit_distance_similarity_of_last_token,correctness\n'
        for old_api in mapping_info:
            for cand in mapping_info[old_api]:
                lines += old_api + ',' + cand + ',' + \
                    str(mapping_info[old_api][cand]['occurrence']) + ',' + \
                    str(mapping_info[old_api][cand]['occurrence_percentage']) + ',' + \
                    str(mapping_info[old_api][cand]['apidoc']) + ',' + \
                    str(mapping_info[old_api][cand]['last_token_similarity']) + ',' + \
                    str(mapping_info[old_api][cand]['correctness'])
                lines += '\n'
        fw.write(lines)

from macros import FQN_MAPPING_PREDICTION_RESULTS_CSV_FILE
def validateFQNMappingPredictionResults(ground_truth_mapping_file=MAPPING_JSON_FILE,
                                        mutation_ground_truth_file=MANUAL_TRAINING_MAPPING_JSON_FILE,
                                        prediction_results_csv_file=FQN_MAPPING_PREDICTION_RESULTS_CSV_FILE):
    with open(prediction_results_csv_file, 'r') as fr:
        lines = fr.readlines()[1:]
    prob_dict = collections.OrderedDict({})
    true_dict = collections.OrderedDict({})
    for i, l in enumerate(lines):
        old_api = l.strip().split(',')[0]
        if old_api not in prob_dict:
            prob_dict[old_api] = []
        cand_api = l.strip().split(',')[1]
        prob = l.strip().split(',')[-1]
        cands = [x[0] for x in prob_dict[old_api]]
        if cand_api not in cands:
            prob_dict[old_api].append((cand_api, prob))
        correctness = l.strip().split(',')[-3]
        if correctness == 'True':
            true_dict[old_api] = cand_api
    # sort by prob
    for old_api in prob_dict:
        prob_dict[old_api] = sorted(prob_dict[old_api], key=lambda x: float(x[1]), reverse=True)
    top_1_prob_dict = collections.OrderedDict({})
    top_3_prob_dict = collections.OrderedDict({})
    top_5_prob_dict = collections.OrderedDict({})
    for old_api in prob_dict:
        top_1_prob_dict[old_api] = prob_dict[old_api][:1]
        top_3_prob_dict[old_api] = prob_dict[old_api][:3]
        top_5_prob_dict[old_api] = prob_dict[old_api][:5]
    print('TOP 1 RESULT: ')
    for old_api in top_1_prob_dict:
        print(old_api + ' -> ' + top_1_prob_dict[old_api][0][0])
    num_of_top_1_hit = 0
    num_of_top_3_hit = 0
    num_of_top_5_hit = 0
    hit_map = collections.OrderedDict(dict.fromkeys(list(prob_dict.keys()), ""))
    for old_api in prob_dict:
        if old_api not in true_dict: # e.g., plotly.plotly
            hit_map[old_api] = 'None'
            continue
        true_cand = true_dict[old_api]
        if true_cand in [x[0] for x in top_1_prob_dict[old_api]]:
            num_of_top_1_hit += 1
            num_of_top_3_hit += 1
            num_of_top_5_hit += 1
            hit_map[old_api] += "top-1"
        elif true_cand in [x[0] for x in top_3_prob_dict[old_api]]:
            num_of_top_3_hit += 1
            num_of_top_5_hit += 1
            hit_map[old_api] += "top-3"
        elif true_cand in [x[0] for x in top_5_prob_dict[old_api]]:
            num_of_top_5_hit += 1
            hit_map[old_api] += "top-5"
        else:
            #print('&&& ' + true_cand)
            #print(prob_dict[old_api])
            keys = [x[0] for x in prob_dict[old_api]]
            if true_cand not in keys:
                hit_map[old_api] += "bad-INF"
            else:
                hit_map[old_api] += "bad-" + str(keys.index(true_cand))
    print('Total: ' + str(len(prob_dict)))
    print('TOP 1 HIT: ' + str(num_of_top_1_hit))
    print('TOP 3 HIT: ' + str(num_of_top_3_hit))
    print('TOP 5 HIT: ' + str(num_of_top_5_hit))
    for old_api in hit_map:
        print(old_api, hit_map[old_api])

def computeMetricsOfFQNMappingPrediction(prediction_results_csv_file=FQN_MAPPING_PREDICTION_RESULTS_CSV_FILE):
    with open(prediction_results_csv_file, 'r') as fr:
        lines = fr.readlines()
    old_api_to_num_of_data_points_map = collections.OrderedDict({})
    for i, l in enumerate(lines):
        old_api = l.strip().split(',')[0]
        if old_api not in old_api_to_num_of_data_points_map:
            old_api_to_num_of_data_points_map[old_api] = 1
        else:
            old_api_to_num_of_data_points_map[old_api] += 1
    num_of_data_points_list = list(old_api_to_num_of_data_points_map.values())
    max_num_of_data_points = max(num_of_data_points_list)
    median_num_of_data_points = statistics.median(num_of_data_points_list)
    mean_num_of_data_points = round(statistics.mean(num_of_data_points_list), 2)
    print('MAX NUM: ' + str(max_num_of_data_points))
    print('MEDIAN NUM: ' + str(median_num_of_data_points))
    print('MEAN NUM: ' + str(mean_num_of_data_points))

def extractErrorMsgFromLogFile(log_file):
    with open(log_file, 'r') as fr:
        lines = fr.readlines()
    error_msg = 'PASS'
    for i, l in enumerate(lines):
        if 'Error: ' in l:
            if l.strip().split('Error:')[-1] == '':
                error_msg = l.strip() + ' '
                for j in range(i+1, len(lines)):
                    error_msg += lines[j].strip() + ' '
            else:
                error_msg = l.strip()
    return error_msg

def generateErrorMsgDataJSON(subjects_file=SUBJECTS_FILE,
                             results_dir=RESULTS_DIR,
                             our_results_dir=OUR_RESULTS_DIR,
                             error_msg_data_json_file=ERROR_MSG_DATA_JSON):
    with open(subjects_file, 'r') as fr:
        lines = fr.readlines()
    subjects = []
    for i, l in enumerate(lines):
        subjects.append(l.strip())
    error_msg_data = collections.OrderedDict({})
    for s in subjects:
        error_msg_data[s] = collections.OrderedDict({})
        project = s.split('/')[0]
        notebook = s.split('/')[1]
        old_log_file = results_dir + '/' + project + '/' + notebook + '.old.log'
        first_err_msg = extractErrorMsgFromLogFile(old_log_file)
        new_log_file = results_dir + '/' + project + '/' + notebook + '.new.log'
        second_error_msg = extractErrorMsgFromLogFile(new_log_file)
        error_msg_data[s]['iter_1'] = first_err_msg
        error_msg_data[s]['iter_2'] = second_error_msg
    with open(error_msg_data_json_file, 'w') as fw:
        json.dump(error_msg_data, fw, indent=2)

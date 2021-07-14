import os
import sys
import json
import shutil
import collections
import argparse
import time
import subprocess as sub

# macros
from macros import SUBJECTS_FILE
from macros import API_FQN_MAPPING_JSON_FILE
from macros import GITHUB_ONLY_STRATEGY_API_FQN_MAPPING_JSON_FILE
from macros import APIDOC_ONLY_STRATEGY_API_FQN_MAPPING_JSON_FILE
from macros import GITHUB_NAIVE_STRATEGY_API_FQN_MAPPING_JSON_FILE
from macros import APIDOC_NAIVE_STRATEGY_API_FQN_MAPPING_JSON_FILE
from macros import CONVERTED_NOTEBOOKS_DIR
from macros import FIXED_NOTEBOOKS_DIR
from macros import RESULTS_DIR
from macros import PATCHES_DIR

from base_migration import runBaseLineMigration
from ast_migration import runASTMigration
from api_doc_and_err_msgs_migration import runAPIDocAndErrMsgsMigrationPipeline
from api_doc_and_err_msgs_migration import runAPIDocAndErrMsgsMigrationPipelineIteration

from mapping import runBuildMapping
from mapping import runMineCommits
from mapping import runValidateMapping

from module_distribution_collector import collectErrorNotebooksModuleDistribution
from module_distribution_collector import collectPassNotebooksModuleDistribution

from features import generateErrorMsgDataJSON
from features import validateFQNMappingPredictionResults
from features import computeMetricsOfFQNMappingPrediction

from mutation import applyOneNameMutationOnOneNotebook
from mutation import applyOneArgMutationOnOneNotebook
from mutation import genNameMutationTemplatesFromTrainingData
from mutation import genArgMutationTemplatesFromTrainingData
from mutation import runNameMutationUntilKErrorsPerAPI
from mutation import runArgMutationUntilKErrorsPerAPI
from mutation import postProcessDebugInfo
from mutation import getMostFrequentUsedAPIsInKagglePassingNotebooks
from mutation import showNameMutationResults
from mutation import showArgMutationResults
from mutation import genErrMsgToRepairActionCSV
from mutation import printNameMappingWeNeedMine

from mining import runBuildMappingUsingGitHubForCreatingTrainingSet
from mining import runBuildMappingUsingManualArrowFile

from relancer import runRelancer_OneCase

def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='migration mode', required=False)
    parser.add_argument('--project', help='Select project', required=False)
    parser.add_argument('--notebook', help='Select notebook', required=False)
    parser.add_argument('--apidoc', help='Use API doc', action='store_true', required=False)
    parser.add_argument('--github', help='Use GitHub mining', action='store_true', required=False)
    parser.add_argument('--strategy', help='strategy', required=False)
    parser.add_argument('--mutate-name', help='Mutate name', action='store_true', required=False)
    parser.add_argument('--mutate-arg', help='Mutate arg', action='store_true', required=False)
    parser.add_argument('--api', help='API', required=False)
    if len(argv) == 0:
        parser.print_help()
        exit(1)
    opts = parser.parse_args(argv)
    return opts

# --- Deprecated ---
def runAPIMigrationExp_OneCase(mode, project, notebook, strategy,
                               mapping_json_file=API_FQN_MAPPING_JSON_FILE,
                               converted_notebooks_dir=CONVERTED_NOTEBOOKS_DIR,
                               fixed_notebooks_dir=FIXED_NOTEBOOKS_DIR,
                               results_dir=RESULTS_DIR,
                               patches_dir=PATCHES_DIR):
    if strategy == 'github_only':
        mapping_json_file = GITHUB_ONLY_STRATEGY_API_FQN_MAPPING_JSON_FILE
        dirs_suffix = '-github_only'
    elif strategy == 'apidoc_only':
        mapping_json_file = APIDOC_ONLY_STRATEGY_API_FQN_MAPPING_JSON_FILE
        dirs_suffix = '-apidoc_only'
    elif strategy == 'github_naive':
        mapping_json_file = GITHUB_NAIVE_STRATEGY_API_FQN_MAPPING_JSON_FILE
        dirs_suffix = '-github_naive'
    elif strategy == 'apidoc_naive':
        mapping_json_file = APIDOC_NAIVE_STRATEGY_API_FQN_MAPPING_JSON_FILE
        dirs_suffix = '-apidoc_naive'
    else:
        dirs_suffix = ''
    fixed_notebooks_dir += dirs_suffix
    results_dir += dirs_suffix
    patches_dir += dirs_suffix
    # api mapping: old -> new
    with open(mapping_json_file, 'r') as fr:
        api_mapping = json.load(fr, object_pairs_hook=collections.OrderedDict)
    print('--- Running ' + project + '/' + notebook)
    old_file = converted_notebooks_dir + '/' + project + '/' + notebook + '.py'
    new_file = fixed_notebooks_dir + '/' + project + '/' + notebook + '.py'
    if not os.path.isdir(fixed_notebooks_dir + '/' + project):
        os.makedirs(fixed_notebooks_dir + '/' + project)
    if mode == 'baseline':
        shutil.copyfile(old_file, new_file)
        runBaseLineMigration(project, notebook, api_mapping, new_file)
    elif mode == 'ast':
        start_time = time.time()
        shutil.copyfile(old_file, new_file)
        runASTMigration(project, notebook, api_mapping, new_file)
        cwd = os.getcwd()
        # save patches
        patch_output_dir = patches_dir + '/' + project
        if not os.path.isdir(patch_output_dir):
            os.makedirs(patch_output_dir)
        sub.run('diff -u ' + old_file + ' ' + new_file, shell=True,
                stdout=open(patch_output_dir + '/' + notebook + '.patch', 'w'),
                stderr=sub.STDOUT)
        # run buggy
        os.chdir(converted_notebooks_dir + '/' + project)
        output_dir = results_dir + '/' + project
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        buggy_log_file = output_dir + '/' + notebook + '.old.log'
        sub.run('python ' + notebook + '.py', shell=True,
                stdout=open(buggy_log_file, 'w'), stderr=sub.STDOUT)
        buggy_run_end_time = time.time()
        failed_repair_duration = buggy_run_end_time - start_time
        failed_repair_time_log_file = output_dir + '/' + notebook + '.buggy.time'
        with open(failed_repair_time_log_file, 'w') as fw:
            fw.write('[FAILED REPAIR EXEC TIME]: ' + str(failed_repair_duration))
        # run fixed
        os.chdir(fixed_notebooks_dir + '/' + project)
        output_dir = results_dir + '/' + project
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        fixed_log_file = output_dir + '/' + notebook + '.new.log'
        sub.run('python ' + notebook + '.py', shell=True,
                stdout=open(fixed_log_file, 'w'), stderr=sub.STDOUT)
        os.chdir(cwd)
        end_time = time.time()
        duration = end_time - start_time
        time_log_file = output_dir + '/' + notebook + '.time'
        with open(time_log_file, 'w') as fw:
            fw.write('[REPAIR EXEC TIME]: ' + str(duration))
    elif mode == 'api_doc_and_err_msgs':
        # fix api fqn
        #runASTMigration(project, notebook, api_mapping, new_file)
        # fix args
        runAPIDocAndErrMsgsMigrationPipeline(project, notebook, api_mapping, old_file, new_file)
    elif mode == 'api_doc_and_err_msgs_iteration':
        runAPIDocAndErrMsgsMigrationPipelineIteration(project, notebook, api_mapping, old_file, new_file)

# --- Deprecated ---
def runAPIMigrationExp_AllCases(mode, subjects_file=SUBJECTS_FILE):
    # subjects
    subjects = []
    with open(subjects_file, 'r') as fr:
        lines = fr.readlines()
    for i, l in enumerate(lines):
        subjects.append(l.strip())
    # run migration
    for s in subjects:
        project = s.split('/')[0]
        notebook = s.split('/')[1]
        runAPIMigrationExp_OneCase(mode, project, notebook)


if __name__ == '__main__':
    opts = parseArgs(sys.argv[1:])
    mode = opts.mode  # baseline, ast
    if mode == 'mine':
        runMineCommits()
        exit(0)
    elif mode == 'mapping':
        apidoc = True if opts.apidoc else False
        github = True if opts.github else False
        runBuildMapping(apidoc, github, opts.strategy)
        exit(0)
    elif mode == 'validate-mapping':
        runValidateMapping()
        exit(0)
    elif mode == 'module-distribution':
        collectErrorNotebooksModuleDistribution()
        #collectPassNotebooksModuleDistribution()
        exit(0)
    elif mode == 'gen-error-msg-data-json':
        generateErrorMsgDataJSON()
        exit(0)
    elif mode == 'mutation':
        project = opts.project
        notebook = opts.notebook
        api = opts.api
        if project and notebook and api:
            if opts.mutate_name:
                applyOneNameMutationOnOneNotebook(api, project, notebook)
            elif opts.mutate_arg:
                applyOneArgMutationOnOneNotebook(api, project, notebook)
        else:
            runNameMutationUntilKErrorsPerAPI()
            #runArgMutationUntilKErrorsPerAPI()
            #postProcessDebugInfo()
        exit(0)
    elif mode == 'show-mutation-errors':
        #showNameMutationResults()
        #showArgMutationResults()
        genErrMsgToRepairActionCSV()
        #printNameMappingWeNeedMine()
        exit(0)
    elif mode == 'gen-mutation-templates':
        genNameMutationTemplatesFromTrainingData()
        genArgMutationTemplatesFromTrainingData()
        exit(0)
    elif mode == 'training-set':
        runBuildMappingUsingGitHubForCreatingTrainingSet()  # mine github arg patches
        #runBuildMappingUsingManualArrowFile()  # create training set for mutation
        exit(0)
    elif mode == 'kaggle-api-usage':
        getMostFrequentUsedAPIsInKagglePassingNotebooks()
        exit(0)
    elif mode == 'validate-ml-fqn-mappings':
        validateFQNMappingPredictionResults()
        computeMetricsOfFQNMappingPrediction()
        exit(0)
    elif mode == 'count-deprecated-apis-usage':
        from usage import countDeprecatedAPIUsagesInAllErrorCases
        countDeprecatedAPIUsagesInAllErrorCases()
        exit(0)
    elif mode == 'relancer':
        project = opts.project
        notebook = opts.notebook
        strategy = opts.strategy
        runRelancer_OneCase(project, notebook, strategy)
        exit(0)
    if opts.project and opts.notebook:
        project = opts.project
        notebook = opts.notebook
        runAPIMigrationExp_OneCase(mode, project, notebook, opts.strategy)
    else:
        runAPIMigrationExp_AllCases(mode)

import os

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__)) # Dir of this script

EXP_DIR = SCRIPT_DIR + '/../exp'
# subjects: to fix
CONVERTED_NOTEBOOKS_DIR = EXP_DIR + '/_converted_notebooks'

# fix api fqn
FIXED_NOTEBOOKS_DIR = EXP_DIR + '/fixed_converted_notebooks'
RESULTS_DIR = EXP_DIR + '/_results'
PATCHES_DIR = EXP_DIR + '/patches'

# fix args
OUR_FIXED_NOTEBOOKS_DIR = EXP_DIR + '/our_fixed_converted_notebooks'
OUR_RESULTS_DIR = EXP_DIR + '/our_results'
OUR_PATCHES_DIR = EXP_DIR + '/our_patches'
ERROR_MSG_JSON_FILE = EXP_DIR + '/error_msgs.json'
API_DOC_KNOWLEDGE_JSON_FILE = EXP_DIR + '/api_doc_knowledge.json'

SUBJECTS_FILE = EXP_DIR + '/filtered-subjects.txt'

MAPPING_TXT_FILE = EXP_DIR + '/../mapping-apis/mapping.txt'
MAPPING_JSON_FILE = EXP_DIR + '/mapping.json'

CANNOT_USE_FQN_APIS = ['sklearn.impute.SimpleImputer',
                       'statsmodels.api.OLS',
                       'statsmodels.api.Logit']

# mine mapping
MAPPING_EXP_DIR = SCRIPT_DIR + '/../mapping-apis'
DEPRECATED_APIS_LIST_FILE = EXP_DIR + '/deprecated_apis_list.txt'
ACTION_KEYWORDS = ['update', 'upgrade', 'replace']
MINING_TOKENS = ["9b1b1b941faaa8dbf163adfe68debf5a3d30577f"]
MINING_GITHUB_RESULTS_DIR = MAPPING_EXP_DIR + '/_kaggle_results'
MINING_GITHUB_DOWNLOADS_DIR = MAPPING_EXP_DIR + '/_downloads'
API_FQN_MAPPING_JSON_FILE = MAPPING_EXP_DIR + '/generated-mapping.json'
GITHUB_ONLY_API_FQN_MAPPING_JSON_FILE = MAPPING_EXP_DIR + '/generated-mapping-github.json'
APIDOC_ONLY_API_FQN_MAPPING_JSON_FILE = MAPPING_EXP_DIR + '/generated-mapping-apidoc.json'

GITHUB_ONLY_STRATEGY_API_FQN_MAPPING_JSON_FILE = MAPPING_EXP_DIR + '/github-only-strategy-generated-mapping.json'
APIDOC_ONLY_STRATEGY_API_FQN_MAPPING_JSON_FILE = MAPPING_EXP_DIR + '/apidoc-only-strategy-generated-mapping.json'
GITHUB_NAIVE_STRATEGY_API_FQN_MAPPING_JSON_FILE = MAPPING_EXP_DIR + '/github-naive-strategy-generated-mapping.json'
APIDOC_NAIVE_STRATEGY_API_FQN_MAPPING_JSON_FILE = MAPPING_EXP_DIR + '/apidoc-naive-strategy-generated-mapping.json'

# mine api doc
SKLEARN_API_DOC_FILE = MAPPING_EXP_DIR + '/sklearn-apis-0.23.1.txt'
PANDAS_API_DOC_FILE = MAPPING_EXP_DIR + '/pandas-apis-1.0.5.txt'
TENSORFLOW_API_DOC_FILE = MAPPING_EXP_DIR + '/tensorflow-apis-2.2.0.txt'
STATSMODELS_API_DOC_FILE = MAPPING_EXP_DIR + '/statsmodels-apis-0.11.1.txt'
SEABORN_API_DOC_FILE = MAPPING_EXP_DIR + '/seaborn-apis-0.10.1.txt'
NETWORKX_API_DOC_FILE = MAPPING_EXP_DIR + '/networkx-apis-2.4.txt'
SCIPY_API_DOC_FILE = MAPPING_EXP_DIR + '/scipy-apis-1.5.0.txt'
PLOTLY_API_DOC_FILE = MAPPING_EXP_DIR + '/plotly-apis-4.13.0.txt'
KERAS_API_DOC_FILE = MAPPING_EXP_DIR + '/keras-2.4.3.txt'
NUMPY_API_DOC_FILE = MAPPING_EXP_DIR + '/numpy-apis-1.18.5.txt'
NLTK_API_DOC_FILE = MAPPING_EXP_DIR + '/'
IMBLEARN_API_DOC_FILE = MAPPING_EXP_DIR + '/imblearn-apis-0.7.0.txt'

# features
FEATURES_CSV_FILE = MAPPING_EXP_DIR + '/features.csv'

# module distribution
STUDY_DIR = SCRIPT_DIR + '/../study'
MODULE_DISTRIBUTION_DATASET_FILE = STUDY_DIR + '/dataset.txt'  # top-350, ImportError + ModuleNotFoundError + AttributeError
MODULE_DISTRIBUTION_CONVERTED_NOTEBOOKS_DIR = STUDY_DIR + '/_converted_notebooks'

# error msg clustering
ERROR_MSG_DATA_JSON = SCRIPT_DIR + '/../cluster/error-msg-data.json'

# mutation for creating training data
MUTATION_EXP_DIR = SCRIPT_DIR + '/../mine-actions'
MUTATION_EXP_SUBJECTS_FILE = MUTATION_EXP_DIR + '/result-of-long-run.json'
NAME_MUTATION_TEMPLATE_JSON_FILE = MUTATION_EXP_DIR + '/mutation-name.json'
ARG_MUTATION_TEMPLATE_JSON_FILE = MUTATION_EXP_DIR + '/mutate-arg.json'
MUTATION_EXP_CONVERTED_NOTEBOOKS_DIR = MUTATION_EXP_DIR + '/_converted_notebooks'
NAME_MUTATION_EXP_MUTATED_NOTEBOOKS_DIR = MUTATION_EXP_DIR + '/_mutated_notebooks_name'
NAME_MUTATION_EXP_LOGS_DIR = MUTATION_EXP_DIR + '/_logs_name'
NAME_MUTATION_EXP_PATCHES_DIR = MUTATION_EXP_DIR + '/_patches_name'
ARG_MUTATION_EXP_MUTATED_NOTEBOOKS_DIR = MUTATION_EXP_DIR + '/_mutated_notebooks_args'
ARG_MUTATION_EXP_LOGS_DIR = MUTATION_EXP_DIR + '/_logs_args'
ARG_MUTATION_EXP_PATCHES_DIR = MUTATION_EXP_DIR + '/_patches_args'
JUPYTER_EXP_CONVERTED_NOTEBOOKS_DIR = MUTATION_EXP_DIR + '/../../jupyter/_converted_notebooks'
# mining github for creating err-action training data
MINE_GITHUB_TRAINING_MAPPING_JSON_FILE = MUTATION_EXP_DIR + '/training-mapping.json'
MINE_GITHUB_TRAINING_ARG_MAPPING_JSON_FILE = MUTATION_EXP_DIR + '/training-mapping-args.json'
MANUALLY_VALIDATED_TRAINING_NAME_MAPPING_JSON_FILE = MUTATION_EXP_DIR + '/validated-training-mapping.json'
MANUALLY_VALIDATED_TRAINING_ARG_MAPPING_JSON_FILE = MUTATION_EXP_DIR + '/validated-training-mapping-args.json'
META_INFO_MAPPING_FILE = MUTATION_EXP_DIR + '/mapping-meta-info.json'
TRAINING_NAME_MUTATION_TEMPLATE_JSON_FILE = MUTATION_EXP_DIR + '/training-name-mutation.json'
TRAINING_ARG_MUTATION_TEMPLATE_JSON_FILE = MUTATION_EXP_DIR + '/training-arg-mutation.json'
NAME_MUTATION_RESULTS_INFO_JSON_FILE = MUTATION_EXP_DIR + '/name-mutation-results.json'
ARG_MUTATION_RESULTS_INFO_JSON_FILE = MUTATION_EXP_DIR + '/arg-mutation-results.json'

TIME_OUT_THRESHOLD = 300
DEBUG_INFO_FILE = SCRIPT_DIR + '/debug'

KAGGLE_API_USAGE_INFO_JSON_FILE = MUTATION_EXP_DIR + '/kaggle-api-usage.json'
KAGGLE_API_USAGE_20_LIBS_INFO_JSON_FILE = MUTATION_EXP_DIR + '/kaggle-api-usage-20-libs.json'

ERROR_NOTEBOOKS_IMPORTED_MODULE_DISTRIBUTION_FILE = MUTATION_EXP_DIR + '/module_distribution/error-imported-modules.json'
ERROR_NOTEBOOKS_USED_MODULE_DISTRIBUTION_FILE = MUTATION_EXP_DIR + '/module_distribution/error-used-modules.json'
PASSING_NOTEBOOKS_IMPORTED_MODULE_DISTRIBUTION_FILE = MUTATION_EXP_DIR + '/module_distribution/success-imported-modules.json'
PASSING_NOTEBOOKS_USED_MODULE_DISTRIBUTION_FILE = MUTATION_EXP_DIR + '/module_distribution/success-used-modules.json'
MANUAL_KNOWLEDGE_FILE = MUTATION_EXP_DIR + '/manual_found_deprecated_apis_20_libs.txt'
MANUAL_TRAINING_MAPPING_JSON_FILE = MUTATION_EXP_DIR + '/manual-training-mapping.json'
MANUAL_TRAINING_ARG_MAPPING_JSON_FILE = MUTATION_EXP_DIR + '/manual-training-mapping-args.json'
MANUAL_TRAINING_NAME_MUTATION_TEMPLATE_JSON_FILE = MUTATION_EXP_DIR + '/manual-training-name-mutation.json'
MANUAL_TRAINING_ARG_MUTATION_TEMPLATE_JSON_FILE = MUTATION_EXP_DIR + '/manual-training-arg-mutation.json'
MANUAL_NAME_MUTATION_RESULTS_INFO_JSON_FILE = MUTATION_EXP_DIR + '/manual-name-mutation-results.json'
MANUAL_ARG_MUTATION_RESULTS_INFO_JSON_FILE = MUTATION_EXP_DIR + '/manual-arg-mutation-results.json'
ERROR_MSG_TO_REPAIR_ACTION_TRAIN_CSV_FILE = MUTATION_EXP_DIR + '/error-msg-to-repair-action-TRAIN.csv'
ERROR_MSG_TO_REPAIR_ACTION_TEST_CSV_FILE = MUTATION_EXP_DIR + '/error-msg-to-repair-action-TEST.csv'

FQN_MAPPING_PREDICTION_RESULTS_CSV_FILE = MAPPING_EXP_DIR + '/candidate_ranking.csv'

# RELANCER EXP
ASE_RELANCER_EXP_DIR = SCRIPT_DIR + '/../relancer-exp'
ASE_RELANCER_FQN_MAPPING_JSON = ASE_RELANCER_EXP_DIR + '/relancer_fqns.json'
ASE_RELANCER_GITHUB_FQN_MAPPING_FILE = ASE_RELANCER_EXP_DIR + '/relancer_github_fqns.json'
ASE_RELANCER_APIDOC_FQN_MAPPING_FILE = ASE_RELANCER_EXP_DIR + '/relancer_apidoc_fqns.json'
ASE_ORIGINAL_NOTEBOOKS_DIR = ASE_RELANCER_EXP_DIR + '/original_notebooks'
ASE_ORIGINAL_INPUT_DIR = ASE_RELANCER_EXP_DIR + '/input'  # 500 MB
ASE_FIXED_NOTEBOOKS_DIR = ASE_RELANCER_EXP_DIR + '/fixed_notebooks'
ASE_EXECUTION_LOGS_DIR = ASE_RELANCER_EXP_DIR + '/exec-logs'
ASE_REPAIR_LOGS_DIR = ASE_RELANCER_EXP_DIR + '/fix-logs'
ASE_PATCHES_DIR = ASE_RELANCER_EXP_DIR + '/patches'
ASE_APIDOC_KNOWLEDGE_JSON = ASE_RELANCER_EXP_DIR + '/relancer_apidoc_args.json'
ASE_GITHUB_KNOWLEDGE_JSON = ASE_RELANCER_EXP_DIR + '/relancer_github_args.json'
ASE_GROUND_TRUTH_FIXES_JSON = ASE_RELANCER_EXP_DIR + '/each-notebook-fixes.json'
ASE_CANDIDATE_RANKING_MODEL_OUTPUT_CSV_FILE = ASE_RELANCER_EXP_DIR + '/candidate_ranking.csv'
R_REPAIR_ACTION_MODEL_PREDICTION_CSV = ASE_RELANCER_EXP_DIR + '/error_msgs.csv'

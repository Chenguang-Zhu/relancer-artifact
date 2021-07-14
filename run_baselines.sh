# Create directories to store final and intermediate results

mkdir -p relancer-exp/patches
mkdir -p relancer-exp/fixed_notebooks
mkdir -p relancer-exp/exec-logs

# Run the Relancer_github baseline on all the Jupyter Notebooks used in the evaluation (RQ2)
(
    cd relancer/scripts/full
    ./relancer_github.sh | tee ../../../relancer-exp/batch-log-github.txt
)

# Remove irrelevant files produced by the notebooks
find relancer-exp/fixed_notebooks/ -type f ! -name "*.py" | xargs rm &> /dev/null

# Run the Relancer_doc baseline on all the Jupyter Notebooks used in the evaluation (RQ2)
(
    cd relancer/scripts/full
    ./relancer_doc.sh | tee ../../../relancer-exp/batch-log-apidoc.txt
)

# Remove irrelevant files produced by the notebooks
find relancer-exp/fixed_notebooks/ -type f ! -name "*.py" | xargs rm &> /dev/null

# Run the Relancer_text baseline on all the Jupyter Notebooks used in the evaluation (RQ3)
(
    cd relancer/scripts/full
    ./relancer_text.sh | tee ../../../relancer-exp/batch-log-text.txt
)

# Remove irrelevant files produced by the notebooks
find relancer-exp/fixed_notebooks/ -type f ! -name "*.py" | xargs rm &> /dev/null

# Run the Relancer_random baseline on all the Jupyter Notebooks used in the evaluation (RQ3)
(
    cd relancer/scripts/full
    ./relancer_random.sh | tee ../../../relancer-exp/batch-log-random.txt
)

# Remove irrelevant files produced by the notebooks
find relancer-exp/fixed_notebooks/ -type f ! -name "*.py" | xargs rm &> /dev/null

# Run the Relancer_naive baseline on all the Jupyter Notebooks used in the evaluation (RQ3)
(
    cd relancer/scripts/full
    ./relancer_naive.sh | tee ../../../relancer-exp/batch-log-naive.txt
)

# Remove irrelevant files produced by the notebooks
find relancer-exp/fixed_notebooks/ -type f ! -name "*.py" | xargs rm &> /dev/null


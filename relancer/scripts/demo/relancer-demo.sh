#!/bin/bash

cd ../../

date

time timeout 1800 python main.py --mode relancer --project pavansubhasht_ibm-hr-analytics-attrition-dataset --notebook ibm-hr-prediction-and-data-visualization --strategy relancer
time timeout 1800 python main.py --mode relancer --project ruslankl_mice-protein-expression --notebook behavior-based-on-protein-expression-using-svm --strategy relancer
time timeout 1800 python main.py --mode relancer --project alopez247_pokemon --notebook sklearn-tutorial --strategy relancer
time timeout 1800 python main.py --mode relancer --project ronitf_heart-disease-uci --notebook analyzing-the-heart-disease --strategy relancer
time timeout 1800 python main.py --mode relancer --project neuromusic_avocado-prices --notebook explore-avocados-from-all-sides --strategy relancer
time timeout 1800 python main.py --mode relancer --project aljarah_xAPI-Edu-Data --notebook data-analysis-student-s-behavior --strategy relancer
time timeout 1800 python main.py --mode relancer --project budincsevity_szeged-weather --notebook simple-linear-regression-using-weatherhistory-data --strategy relancer
time timeout 1800 python main.py --mode relancer --project toramky_automobile-dataset --notebook auto-imports-beginner-level-analysis --strategy relancer
time timeout 1800 python main.py --mode relancer --project annavictoria_speed-dating-experiment --notebook does-everyone-like-the-same-people-eda --strategy relancer
time timeout 1800 python main.py --mode relancer --project johndasilva_diabetes --notebook diabetes-classifications --strategy relancer

date

cd -

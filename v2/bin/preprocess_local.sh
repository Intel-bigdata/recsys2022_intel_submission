#bin/bash
set -x

# data desciption
# valid_sessions_31d = valid_sessions_new + valid_purchases_new


python src/preprocess_local.py
# step1: produce train_small_23.txt, train_small_25.txt, train_11-16_month.txt
# step2: produce valid.txt
# step3: produce valid_test.txt
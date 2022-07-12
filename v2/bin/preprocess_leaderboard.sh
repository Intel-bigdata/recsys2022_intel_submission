#bin/bash
set -x



python src/preprocess_leaderboard.py
# step1: produce train_final_small_23.txt, train_final_small_25.txt, train_12-17_month.txt
# step2: produce test.txt, test_final.txt

#bin/bash
# set -x

conda activate serec 

model_dir='/content/drive/MyDrive/models/cat3_valid_pt6m'
log_path=$model_dir'/run.log'
embedding_dim=128
num_layer=1
order=3
item_date_file='valid_item_release_date.pickle'

train_file='train_11-16_month.txt'
train_epoch=4

finetune_file='train_small_25.txt'
finetune_epoch=1

test_file='valid_test.txt'
test_candidate_file='valid_candidate_items.csv'
sub_file='sub_pt_ft.csv'


mkdir -p $model_dir


# Train
python -u src/main.py \
  --dataset-dir ../datasets/dressipi \
  --item-date-file $item_date_file \
  --candidate-file $test_candidate_file
  --model-dir $model_dir \
  --task train \
  --train-dataset $train_file \
  --train-mode 'keep_next' \
  --edge-drop 0.2 \
  --num-layers $num_layer --order $order --embedding-dim $embedding_dim \
  --epochs $train_epoch --num-workers 4 --weight-decay 0 \
  2>&1 | tee -a $log_path


# Finetune
python -u src/main.py \
  --dataset-dir ../datasets/dressipi \
  --item-date-file $item_date_file \
  --candidate-file $test_candidate_file
  --model-dir $model_dir \
  --task train --finetune \
  --train-dataset $finetune_file \
  --num-layers $num_layer --order $order --embedding-dim $embedding_dim \
  --epochs $finetune_epoch --num-workers 4 \
  2>&1 | tee -a $log_path


# evaluate
python -u src/main.py \
  --dataset-dir ../datasets/dressipi \
  --item-date-file $item_date_file \
  --candidate-file $test_candidate_file
  --model-dir $model_dir \
  --task eval \
  --valid-dataset 'train_small_25.txt' \
  --num-layers $num_layer --order $order --embedding-dim $embedding_dim \
  --num-workers 4 \
  2>&1 | tee -a $log_path


# predict
python -u src/main.py \
  --dataset-dir ../datasets/dressipi \
  --item-date-file $item_date_file \
  --model-dir $model_dir \
  --task predict \
  --test-dataset $test_file \
  --candidate-file $test_candidate_file \
  --submission-name $sub_file \
  --num-layers $num_layer --order $order --embedding-dim $embedding_dim \
  --epochs 1 --num-workers 4 \
  2>&1 | tee -a $log_path

data_dir='/path/to/your/data'
        
# predict test
save_path='/path/to/save/prediction'
model_path='/path/to/saved/model'
python -u src/scripts/main.py \
        --save_path  $save_path \
        --model $model_path \
        --dataset-dir $data_dir \
        --predict \
        --valid-session 'test.csv' \
        --valid-purchase ""  \
        --candidate-list 'candidate_items.csv' \
        --topk 5

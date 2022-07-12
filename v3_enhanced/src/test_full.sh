export CUDA_VISIBLE_DEVICES=4
name="leaderboard_v3_enhanced_5month"
folder="../history/leaderboard/${name}"
exclude_ids="29 44"
extra_key="binned_count_item_clicks binned_elapse_to_end"
month=5
order=1
mkdir -p ${folder}

echo python scripts/main_sihg4sr.py --order ${order}  --extra_feat_key ${extra_key} --exclude_feat_id ${exclude_ids} --dataset-dir ../../dressipi/ --train-session train_sessions.csv --train-purchase train_purchases.csv --save_path "${folder}" --train-clickthrough --use-recent-n-month ${month} | tee -a "${folder}/log"
python -u scripts/main_sihg4sr.py --order ${order} --extra_feat_key ${extra_key} --exclude_feat_id ${exclude_ids} --dataset-dir ../../dressipi/ --train-session train_sessions.csv --train-purchase train_purchases.csv --save_path "${folder}" --train-clickthrough --use-recent-n-month ${month} | tee -a "${folder}/log"
ls ${folder}/model_*
if [ "$?" = 2 ]; then
    exit
fi
model_name=`ls ${folder}/model_*_epoch_3.pth | sort -V | tail -1`
echo python scripts/main_sihg4sr.py --extra_feat_key ${extra_key} --exclude_feat_id ${exclude_ids} --order ${order} --dataset-dir ../../dressipi/ --train-session train_sessions.csv --train-purchase train_purchases.csv --save_path "${folder}" --model "${model_name}" --sort-train-data --finetune | tee -a "${folder}/log"
python -u scripts/main_sihg4sr.py --extra_feat_key ${extra_key} --exclude_feat_id ${exclude_ids} --order ${order} --dataset-dir ../../dressipi/ --train-session train_sessions.csv --train-purchase train_purchases.csv --save_path "${folder}" --model "${model_name}" --sort-train-data --finetune | tee -a "${folder}/log"

model_name=`ls ${folder}/model_*_finetune_3.pth | sort -V | tail -1`
echo python scripts/main_sihg4sr.py --extra_feat_key ${extra_key} --exclude_feat_id ${exclude_ids} --order ${order} --dataset-dir ../../dressipi/ --predict-leaderboard --save_path "${folder}" --model "${model_name}" | tee -a "${folder}/log"
python -u scripts/main_sihg4sr.py --extra_feat_key ${extra_key} --exclude_feat_id ${exclude_ids} --order ${order} --dataset-dir ../../dressipi/ --predict-leaderboard --save_path "${folder}" --model "${model_name}" | tee -a "${folder}/log"
mv "${folder}/prediction.csv" "${folder}/${name}_prediction.csv"

echo python scripts/main_sihg4sr.py --extra_feat_key ${extra_key} --exclude_feat_id ${exclude_ids} --order ${order} --dataset-dir ../../dressipi/ --predict-final --save_path "${folder}" --model "${model_name}" | tee -a "${folder}/log"
python -u scripts/main_sihg4sr.py --extra_feat_key ${extra_key} --exclude_feat_id ${exclude_ids} --order ${order} --dataset-dir ../../dressipi/ --predict-final --save_path "${folder}" --model "${model_name}" | tee -a "${folder}/log"
mv "${folder}/prediction.csv" "${folder}/final_${name}_prediction.csv"

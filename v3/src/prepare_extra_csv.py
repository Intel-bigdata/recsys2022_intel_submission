import pandas as pd
import numpy as np

train = pd.read_csv("../../dressipi/train_sessions.csv")
purchase = pd.read_csv("../../dressipi/train_purchases.csv")
all_items = pd.concat([train, purchase], ignore_index=True)

item_clicks = all_items.groupby("item_id", as_index=False).agg(count_item_clicks=('item_id', 'count'))
item_clicks['binned_count_item_clicks'] = pd.cut(item_clicks['count_item_clicks'], [1, 15, 111, 313, 23165]).cat.codes
item_clicks['binned_count_item_clicks'] = item_clicks['binned_count_item_clicks'].astype(int)
item_clicks[['item_id', 'binned_count_item_clicks']].to_csv('item_features_extra.csv', index=False)

kg_df = pd.read_csv("../../dressipi/item_features.csv")
kg_df['feature_category_id'] = kg_df['feature_category_id'].astype("string")
kg_df['feature_value_id'] = kg_df['feature_value_id'].astype("string")
kg_df["feature_merge"] = "f_" + kg_df['feature_category_id'] + "=" + kg_df['feature_value_id']
codes_feature, uniques_feature = pd.factorize(kg_df["feature_merge"])
kg_df["feature"] = pd.Categorical(codes_feature, categories=range(len(uniques_feature)))
kg_df = kg_df.groupby('feature', as_index=False).agg({'feature_category_id': 'max', 'feature_merge': 'max'})
kg_df[['feature', 'feature_category_id', 'feature_merge']].to_csv('categorical_item_features.csv', index=False)

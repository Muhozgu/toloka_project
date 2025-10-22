import pandas as pd
if len(merged) < 5:
kappa_mat.loc[a,b] = np.nan
else:
kappa_mat.loc[a,b] = cohen_kappa_score(merged['label_a'], merged['label_b'])
return kappa_mat




def fleiss_kappa(table):
"""
Implementation of Fleiss' Kappa for m annotators per item.
`table` is a numpy array of shape (n_items, n_categories) with counts per category per item.
"""
N, k = table.shape
n_annotators = table.sum(axis=1)[0]
p = table.sum(axis=0) / (N * n_annotators)
P = ( (table * (table - 1)).sum(axis=1) ) / (n_annotators * (n_annotators - 1))
Pbar = P.mean()
PbarE = (p * p).sum()
kappa = (Pbar - PbarE) / (1 - PbarE)
return kappa




def compute_fleiss_kappa_from_df(df, items=None, label_order=None):
# df has columns item_id, annotator_id, label
if items is None:
items = sorted(df['item_id'].unique())
labels = label_order or sorted(df['label'].unique())
table = np.zeros((len(items), len(labels)), dtype=int)
item_to_idx = {it: i for i, it in enumerate(items)}
label_to_idx = {l: i for i, l in enumerate(labels)}
for _, row in df.iterrows():
i = item_to_idx[row['item_id']]
j = label_to_idx[row['label']]
table[i, j] += 1
return fleiss_kappa(table)




def annotator_statistics(df):
"""Compute simple statistics per annotator: num_labels, unique_items, distribution"""
stats = df.groupby('annotator_id').agg(num_labels=('label','count'),
unique_items=('item_id',pd.Series.nunique))
# label distribution per annotator
label_dist = df.pivot_table(index='annotator_id', columns='label', values='item_id', aggfunc='count', fill_value=0)
return stats, label_dist
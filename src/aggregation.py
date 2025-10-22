import numpy as np

items = sorted(df['item_id'].unique())
annotators = sorted(df['annotator_id'].unique())
labels = labels or sorted(df['label'].unique())
label_to_idx = {l:i for i,l in enumerate(labels)}
n_items = len(items)


# initialize priors uniformly
prior = np.ones(len(labels)) / len(labels)


# initialize annotator confusion matrices to near-identity + noise
annotator_cm = {a: np.eye(len(labels)) * 0.9 + 0.1/len(labels) for a in annotators}


# Build observations per item
obs = defaultdict(list)
for _, row in df.iterrows():
obs[row['item_id']].append((row['annotator_id'], label_to_idx[row['label']]))


# Initialize posterior probabilities per item: proportional to label frequency
post = {item: np.ones(len(labels)) / len(labels) for item in items}


for iteration in range(max_iter):
# E-step: compute posterior over true label for each item
for item in items:
logp = np.log(prior + 1e-12)
for a, lab_idx in obs[item]:
cm = annotator_cm[a]
# multiply by P(annotator labels lab_idx | true = t) -> use cm[:, lab_idx]
logp += np.log(cm[:, lab_idx] + 1e-12)
# normalize
maxlog = np.max(logp)
p = np.exp(logp - maxlog)
p = p / p.sum()
post[item] = p


# M-step: update priors and annotator confusion matrices
new_prior = np.zeros_like(prior)
new_cm = {a: np.zeros((len(labels), len(labels))) for a in annotators}


for item in items:
p = post[item]
new_prior += p
for a, lab_idx in obs[item]:
# responsibility: add p to row true label and column observed label
for true_idx in range(len(labels)):
new_cm[a][true_idx, lab_idx] += p[true_idx]


# normalize
new_prior = new_prior / new_prior.sum()
for a in annotators:
row_sums = new_cm[a].sum(axis=1, keepdims=True)
# avoid division by zero
row_sums[row_sums == 0] = 1
new_cm[a] = new_cm[a] / row_sums


# check convergence
max_diff = max((np.abs(new_prior - prior).max(),) + tuple(np.abs(new_cm[a] - annotator_cm[a]).max() for a in annotators))
prior = new_prior
annotator_cm = new_cm
if max_diff < tol:
break


# convert post to long DataFrame
rows = []
for item in items:
for i, lab in enumerate(labels):
rows.append({'item_id': item, 'label': labels[i], 'prob': float(post[item][i])})
post_df = pd.DataFrame(rows)
annotator_cm_df = {a: pd.DataFrame(annotator_cm[a], index=labels, columns=labels) for a in annotators}
return post_df, annotator_cm_df
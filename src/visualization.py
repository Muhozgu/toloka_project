import matplotlib.pyplot as plt
plt.tight_layout()
plt.savefig(outpath)
plt.close()




def plot_annotator_heatmap(kappa_df, outpath="results/plots/annotator_kappa_heatmap.png"):
plt.figure(figsize=(8,6))
sns.heatmap(kappa_df.astype(float), annot=True, fmt='.2f', cmap='vlag', center=0)
plt.title('Pairwise Cohen\'s Kappa between annotators')
plt.tight_layout()
plt.savefig(outpath)
plt.close()


---


### FILE: src/main.py


from src.utils import ensure_dirs, save_annotations_csv, generate_synthetic_annotations, load_annotations_csv
from src.metrics import pairwise_cohen_kappa, compute_fleiss_kappa_from_df, annotator_statistics
from src.aggregation import majority_vote, weighted_vote, dawid_skene
from src.visualization import plot_label_distribution, plot_annotator_heatmap


import pandas as pd




def main():
ensure_dirs()


print("Generating synthetic annotations...")
df = generate_synthetic_annotations(n_items=500, n_annotators=10)
save_annotations_csv(df)


print("Computing annotator statistics...")
stats, label_dist = annotator_statistics(df)
stats.to_csv('results/annotator_stats.csv')
label_dist.to_csv('results/annotator_label_dist.csv')


print("Agreement metrics: pairwise Cohen's kappa...")
kappa_df = pairwise_cohen_kappa(df)
kappa_df.to_csv('results/pairwise_kappa.csv')


print("Fleiss' Kappa...")
fk = compute_fleiss_kappa_from_df(df)
with open('results/fleiss_kappa.txt', 'w') as f:
f.write(str(fk))


print("Visualizations...")
plot_label_distribution(df)
plot_annotator_heatmap(kappa_df)


print("Aggregations: majority vote...")
maj = majority_vote(df)
maj.to_csv('results/aggregated_majority.csv', index=False)


print("Running Dawid-Skene EM...")
post_df, annotator_cm = dawid_skene(df)
post_df.to_csv('results/dawid_skene_posteriors.csv', index=False)
# save one annotator confusion matrix example
list(annotator_cm.items())[0][1].to_csv('results/annotator_cm_example.csv')


# combine to a final label by picking highest posterior
final = post_df.loc[post_df.groupby('item_id')['prob'].idxmax()][['item_id','label']].rename(columns={'label':'dawid_label'})
final.to_csv('results/aggregated_dawid.csv', index=False)


print("Done. Results in ./results/")




if __name__ == '__main__':
main()
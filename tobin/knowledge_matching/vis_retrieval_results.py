




# Now let's plot the results and save the data
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming rounding_result_df is already defined and contains the necessary data
fig, ax = plt.subplots()

# Group data by 'rounding_type' and plot
for key, grp in rounding_result_df.groupby(['rounding_type']):
    ax = grp.plot(ax=ax, kind='line', x='diff', y='Recall@10', label=key)

sns.lineplot(x='diff', y='Recall@10', hue='rounding_type', style='round_corpus', data=rounding_result_df)

plt.title('Recall@10 by Rounding Parameter and Type')
plt.xlabel('Rounding Parameter (rounding_p)')
plt.ylabel('Recall@10')
plt.legend(title='Rounding Type')
plt.grid(True)
plt.show()





# Separate task: let's visualise what matching looks like across the datasets. We're going to take the embeddings of the dataset and plot them in 2-D
import torch
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
import matplotlib.patches as patches


path_query_embeddings = f'../datasets/{DATASET}/query_embeddings.pt'
path_corpus_embeddings = f'../datasets/{DATASET}/corpus_embeddings.pt'

query_embeddings = torch.load(path_query_embeddings)
corpus_embeddings = torch.load(path_corpus_embeddings)

# PCA into two dimensions
pca = PCA(n_components=2)
corpus_embeddings_2d = pca.fit_transform(corpus_embeddings)
query_embeddings_2d = pca.transform(query_embeddings)

umap_reducer = umap.UMAP(n_components=2)
corpus_embeddings_2d = umap_reducer.fit_transform(corpus_embeddings.numpy()) 
query_embeddings_2d = umap_reducer.transform(query_embeddings.numpy())


# Plotting the 2D embeddings side by side
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot for query embeddings
axs[0].scatter(query_embeddings_2d[:, 0], query_embeddings_2d[:, 1], alpha=0.5)
axs[0].set_xticks([])
axs[0].set_yticks([])
axs[0].grid(True)
axs[0].set_title('Query Embeddings', fontsize=15)

# Plot for corpus embeddings
axs[1].scatter(corpus_embeddings_2d[:, 0], corpus_embeddings_2d[:, 1], alpha=0.5)
axs[1].tick_params(axis='both', which='both')  # Remove ticks
axs[1].set_xticks([])
axs[1].set_yticks([])
axs[1].grid(True)
axs[1].set_title('Corpus Embeddings', fontsize=15)


def draw_arrow(x_left, y_left, x_right, y_right, width, height):
    # Draw lines connecting the corners of the boxes directly on the figure
    fig.add_artist(plt.Rectangle((x_left, y_left), width, height, linewidth=2, edgecolor='r', facecolor='none'))
    fig.add_artist(plt.Rectangle((x_right, y_right), width, height, linewidth=2, edgecolor='r', facecolor='none'))
    # Calculate the middle points for the lines
    mid_x_left = x_left + width
    mid_y_left = (y_left + y_left + height) / 2
    mid_x_right = x_right
    mid_y_right = (y_right + y_right + height) / 2


    # Create a FancyArrowPatch with double-headed arrow
    arrow = patches.FancyArrowPatch(
        (mid_x_left, mid_y_left),
        (mid_x_right, mid_y_right),
        arrowstyle='<->',
        mutation_scale=20,
        lw=3,
        color='red'
    )

    fig.add_artist(arrow)

# Define the coordinates for the rectangles in data space
x_left, y_left, x_right, y_right, width, height = 0.1, 0.2, 0.61, 0.2, 0.05, 0.1
draw_arrow(x_left, y_left, x_right, y_right, width, height)

x_left, y_left, x_right, y_right, width, height = 0.3, 0.6, 0.81, 0.6, 0.05, 0.1
draw_arrow(x_left, y_left, x_right, y_right, width, height)


plt.tight_layout()
plt.savefig(f'../results/PSI_matching_figure.png', dpi=500)
plt.show()






# # Define the coordinates for the rectangles
# x_left, y_left, width, height = 0.2, 0.2, 0.6, 0.6
# x_right, y_right, width, height = 0.2, 0.2, 0.6, 0.6

# # Drawing a box on the left plot and a corresponding box on the right plot
# rect_left = plt.Rectangle((x_left, y_left), width, height, linewidth=1, edgecolor='r', facecolor='none')
# rect_right = plt.Rectangle((x_right, y_right), width, height, linewidth=1, edgecolor='r', facecolor='none')

# axs[0].add_patch(rect_left)
# axs[1].add_patch(rect_right)

# # Draw lines connecting the corners of the boxes
# for i in range(2):
#     for j in range(2):
#         left_corner = (x_left + i * width, y_left + j * height)
#         right_corner = (x_right + i * width, y_right + j * height)

#         transFigure = fig.transFigure.inverted()
#         coord1 = transFigure.transform(axs[0].transData.transform(left_corner))
#         coord2 = transFigure.transform(axs[1].transData.transform(right_corner))

#         line = plt.Line2D((coord1[0], coord2[0]), (coord1[1], coord2[1]), transform=fig.transFigure, color="red")
#         fig.lines.append(line)

# plt.tight_layout()

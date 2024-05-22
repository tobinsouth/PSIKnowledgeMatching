"""This file will import from the standard library in tools.py where we having working rounding and noise functions. Run `test_embeddings_and_rounding` to ensure the functions are working as expected. 

Then we will run a series of tests here to see how the embeddings are affected by noise and how well the inversion works. The end result will be a publication ready figure. 

Date: Apr 24, 2024
Author: Tobin
"""

from tools import *
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# %% First we'll work with a single sentence to test and provide an example
text_to_embed = ['This is an example sentence for LLM embedding and reconstruction.']

# Let's establish a baseline
embeddings = embed_text(text_to_embed)
if USE_INVERTER:
    inverted = invert_embedding(embeddings)
    inverted_embedding = embed_text(inverted)
    baseline_cos_diff = torch.cosine_similarity(embeddings, inverted_embedding)


# Now we'll loop through and add some noise to the embeddings

embedding_range = [embeddings.min().item(), embeddings.max().item()]
cosine_diffs_pre_invert = {}
cosine_diffs_post_invert = {}
inversions = {}
diff_ranges = {}
n_steps = 100
steps_range = range(2, n_steps)

for i in tqdm(steps_range):
    edges, diff = get_edges(i, embedding_range)
    rounded = fakeround(edges, embeddings)
    gaussed = gauss_noise(embeddings, diff)
    uniformed = uniform_noise(embeddings, diff)

    diff_ranges[i] = diff

    cosine_diffs_pre_invert[i] = {'rounded': torch.cosine_similarity(embeddings, rounded),
                                  'gaussed': torch.cosine_similarity(embeddings, gaussed),
                                  'uniformed': torch.cosine_similarity(embeddings, uniformed)}
    
    if USE_INVERTER:
        inverted_rounded = invert_embedding(rounded)
        inverted_gaussed = invert_embedding(gaussed)
        inverted_uniformed = invert_embedding(uniformed)

        inversions[i] = {'rounded': inverted_rounded,
                    'gaussed': inverted_gaussed,
                    'uniformed': inverted_uniformed}
        

        inv_rounded_embedding = embed_text(inverted_rounded)
        inv_gaussed_embedding = embed_text(inverted_gaussed)
        inv_uniformed_embedding = embed_text(inverted_uniformed)

        cosine_diffs_post_invert[i] = {'rounded': torch.cosine_similarity(embeddings, inv_rounded_embedding),
                                        'gaussed': torch.cosine_similarity(embeddings, inv_gaussed_embedding),
                                        'uniformed': torch.cosine_similarity(embeddings, inv_uniformed_embedding)}
    

# Make the figure
x = [diff_ranges[i]/4 for i in steps_range]
x = [_x/(embedding_range[1] - embedding_range[0]) for _x in x]
y_rounded_pre = [cosine_diffs_pre_invert[i]['rounded'].item() for i in steps_range]
y_gaussed_pre = [cosine_diffs_pre_invert[i]['gaussed'].item() for i in steps_range]
y_uniformed_pre = [cosine_diffs_pre_invert[i]['uniformed'].item() for i in steps_range]

fig, ax = plt.subplots()

ax.plot(x, y_rounded_pre, linestyle='--', alpha=0.5, c = 'r')
ax.plot(x, y_gaussed_pre, linestyle='--', alpha=0.5, c = 'g')
ax.plot(x, y_uniformed_pre,  linestyle='--', alpha=0.5, c = 'b')


if USE_INVERTER:
    # Print results
    print("Original text", text_to_embed)
    print("Basic inversion", inverted)
    print("Inversions for increasing noise")
    for i in steps_range:
        print(f"Step {i}")
        print("Rounded:", inversions[i]['rounded'])
        print("Gaussed:", inversions[i]['gaussed'])
        print("Uniformed:", inversions[i]['uniformed'])

    y_rounded = [cosine_diffs_post_invert[i]['rounded'].item() for i in steps_range]
    y_gaussed = [cosine_diffs_post_invert[i]['gaussed'].item() for i in steps_range]
    y_uniformed = [cosine_diffs_post_invert[i]['uniformed'].item() for i in steps_range]
    y_baseline = baseline_cos_diff.item()


    ax.axhline(y_baseline, color='k', linestyle='dashdot', label='Baseline ')

    ax.plot(x, y_rounded, label='Rounded', c = 'r')
    ax.plot(x, y_gaussed, label='Gauss Noise', c = 'g')
    ax.plot(x, y_uniformed, label='Uniform Noise', c = 'b')

ax.set_xlabel('Mean embedding noise (% of embedding range)')
ax.set_ylabel('Cosine similarity with original embedding')

ax.legend()
h,l = ax.get_legend_handles_labels()
custom_lines = [Line2D([0], [0], color='k', linestyle='--', alpha=0.5), Line2D([0], [0], linestyle='-', color='k', )]
ax.legend(h+custom_lines, l+['Pre Inversion (--)', 'Post Inversion (-)'], title="Modification type", loc='lower left')


# %% Part 2. Now we'll work with multiple sentences to test and provide an example

# The goal here is to do the same as above but will a whole bunch of sentences and then average over the cosine similarities. 








import pandas as pd
import os
import numpy as np

import seaborn as sns; sns.set()
sns.set_style("ticks")
import matplotlib.pyplot as plt

import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, help='path where predictions of any of the experiments are stored')
parser.add_argument('--output_path', type=str, help='path where input cohorts are stored')

args = parser.parse_args()

save_plots=True

std_frame = pd.read_csv(args.input_file)

sns.set()
sns.set_style("ticks")
custom_palette = ["red", "hotpink", "darkgreen", "mediumseagreen", "lightskyblue", "dodgerblue", "blue", "black"]
sns.set_palette(custom_palette)

g=sns.relplot(data = std_frame,
                x = 'IGSD in TCE',
                y = 'IGSD in error rates',
                col='t',
                kind = 'scatter',
                hue='Model',
                style='Metric',
                s=100,
                facet_kws= {'sharey': False, 'sharex': False, 'margin_titles': True},
             )

(g.set_xlabels(fontsize=20)
 .set_ylabels(fontsize=20)
  .set_titles(fontsize=20)
  .set_yticklabels(fontsize=15)
  .set_xticklabels(rotation=45, fontsize=15)
 
)
axes = g.axes.flatten()
axes[0].set_title(label="t = 7.5%",fontsize=20)
axes[1].set_title(label="t = 20%",fontsize=20)

plt.setp(g._legend.get_texts(), fontsize=16)

g.fig.subplots_adjust ( wspace=0.25)
for lh in g._legend.legendHandles: 
    lh.set_alpha(1)
    lh._sizes = [100] 
g.fig.set_size_inches(14,6)
g.savefig(os.path.join(args.output_path, 'fig4.jpeg'), dpi=700)
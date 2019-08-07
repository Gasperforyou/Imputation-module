import Orange.data
from Orange.data import Table
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Load
filename = "../data/ccp_normCounts_mESCquartz.counts.cycle_genes.csv"
dataset = Orange.data.Table(filename)

a = (dataset.X)
b = []
for x in a:
    b.append((x==0).sum())

fig, axs = plt.subplots(1, 1)

axs.hist(b)
axs.set_title('Porazdelitev ničel po celicah', fontsize=20)
axs.set_xlabel('Število ničel', fontsize=20)
axs.set_ylabel('Število celic', fontsize=20)

fig.tight_layout()
fig.savefig('poCelicah.png')

a = (dataset.X.T)
b = []
for x in a:
    b.append((x==0).sum())

fig, axs = plt.subplots(1, 1)

axs.hist(b)
axs.set_title('Porazdelitev ničel po genih', fontsize=20)
axs.set_xlabel('Število ničel', fontsize=20)
axs.set_ylabel('Število genov', fontsize=20)

fig.tight_layout()
fig.savefig('poGenih.png')

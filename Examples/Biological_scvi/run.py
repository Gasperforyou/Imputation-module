import sys
sys.path.append("../..")
import scimpute
import Orange.data
from Orange.data import Table
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# how to load files into numpy
# file_inflated = np.genfromtxt('biological_inflated.csv', delimiter=',')

# Load
filename = "../../data/ccp_normCounts_mESCquartz.counts.cycle_genes.csv"
dataset = Orange.data.Table(filename)

# Izracunaj vse potrebno
dat, mas, zero = scimpute.zero_inflate(dataset.X)
sc = scimpute.ScImpute(dat)
res = sc.scvi()
cor, data = sc.compare(dataset.X, mas, zero)

# Plotaj vse potrebno
# Primerjava bioloskih podatkov z imputiranimi vrednostmi
fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
c = ax0.pcolormesh(dataset.X, norm=colors.LogNorm(vmin=np.amin(dataset.X)+1, vmax=np.amax(dataset.X)), cmap=plt.get_cmap("binary"))
fig.colorbar(c, ax=ax0)
ax0.set_title('Biološki podatki')

c = ax1.pcolormesh(res,norm=colors.LogNorm(vmin=np.amin(res)+1, vmax=np.amax(res)), cmap=plt.get_cmap("binary"))
ax1.set_title('Imputirani biološki podatki')
fig.colorbar(c, ax=ax1)

c = ax2.pcolormesh(dataset.X-res, norm=colors.LogNorm(vmin=np.amin(dataset.X)+1, vmax=np.amax(dataset.X)), cmap=plt.get_cmap("binary"))
ax2.set_title('Razlika med biološkimi podatki in imputacijo')
fig.colorbar(c, ax=ax2)

fig.tight_layout()
fig.savefig('biological_and_imputed.png')



# histogrami matrik
fig, (ax0, ax1, ax2) = plt.subplots(3, 1)

ax0.hist(dataset.X.flatten(), range=(np.amin(dataset.X), 400), bins = 20)
ax0.set_title('Histogram bioloških podatkov')

ax1.hist(res.flatten(), range=(np.amin(dataset.X), 10), bins = 20)
ax1.set_title('Histogram imputiranih podatkov')

ax2.hist((dataset.X-res).flatten(), range=(-100, 300), bins = 20)
ax2.set_title('Histogram razlike med biološkimi in imputiranimi podatki')
ax0.set_xlabel('Ekspresija genov')
ax1.set_xlabel('Ekspresija genov')
ax2.set_xlabel('Ekspresija genov')
fig.tight_layout()
fig.savefig('histogram_matrik.png')

# Histogrami korelacij.
fig, (axs1, axs2) = plt.subplots(2, 2)

axs1[0].hist(data[0])
axs1[0].set_title('Korelacija po profilih celic')
axs1[1].hist(data[1])
axs1[1].set_title('Korelacija po profilih genov')
axs2[0].hist(data[2])
axs2[0].set_title('Po maskiranih profilih celic')
axs2[1].hist(data[3])
axs2[1].set_title('Po maskiranih profilih genov')
fig.tight_layout()
fig.savefig('vrstice_stolpci.png')

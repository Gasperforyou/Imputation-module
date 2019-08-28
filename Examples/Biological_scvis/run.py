import sys
sys.path.append("..\..")
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
sc = scimpute.ScImpute(dataset.X)
res = sc.scvis(dat)
cor, data = sc.compare_embedded(res)


print(cor)
razlika1 = []
razlika2 = []
for x in range(len(data[0])):
    if(data[0][x]>0.7 or data[0][x]<-0.7):
        razlika1.append(data[1][x]-data[0][x])
    if(data[0][x]<0.3 and data[0][x]>-0.3):
        razlika2.append(data[1][x]-data[0][x])



# Plotaj vse potrebno
# Primerjava bioloskih podatkov z imputiranimi vrednostmi
fig, (ax0, ax1) = plt.subplots(2, 1)
c = ax0.pcolormesh(res[0], cmap=plt.get_cmap("binary"))
fig.colorbar(c, ax=ax0)
ax0.set_title('Latentni prostor iz učnih podatkov')
ax0.set_xlabel('Geni', fontsize=12)
ax0.set_ylabel('Celice', fontsize=12)
c = ax1.pcolormesh(res[1], cmap=plt.get_cmap("binary"))
ax1.set_title('Latentni prostor iz testnih podatkov')
ax1.set_xlabel('Geni', fontsize=12)
ax1.set_ylabel('Celice', fontsize=12)
fig.colorbar(c, ax=ax1)

fig.tight_layout()
fig.savefig('Latent1_latent2.png')


# Histogrami korelacij.

fig, axs = plt.subplots(2, 1)

axs[0].hist(razlika1)
axs[0].set_title('Razlika parov korelacij večjih od 0.7 in manjših od -0.7')
axs[1].hist(razlika2)
axs[1].set_title('Razlika parov korelacij manjših od 0.3 in večjih od -0.3')
fig.tight_layout()
fig.savefig('vrstice_stolpci.png')

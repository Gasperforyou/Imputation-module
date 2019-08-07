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
    if(data[0][x]<0.3 or data[0][x]>-0.3):
        razlika2.append(data[1][x]-data[0][x])

# Plotaj vse potrebno
# Primerjava bioloskih podatkov z imputiranimi vrednostmi
fig, (ax0, ax1) = plt.subplots(2, 1)
c = ax0.pcolormesh(res[0], cmap=plt.get_cmap("binary"))
fig.colorbar(c, ax=ax0)
ax0.set_title('Latentni prostor iz učnih podatkov')

c = ax1.pcolormesh(res[1], cmap=plt.get_cmap("binary"))
ax1.set_title('Latentni prostor iz testnih podatkov')
fig.colorbar(c, ax=ax1)

fig.tight_layout()
fig.savefig('Latent1_latent2.png')


# Histogrami korelacij.

fig, axs = plt.subplots(2, 1)

axs[0].hist(razlika1)
axs[0].set_title('Korelacija večja od 0.7 in manjša od -0.7')
axs[1].hist(razlika2)
axs[1].set_title('Korelacija manjša od 0.3 in večja od -0.3')
fig.tight_layout()
fig.savefig('vrstice_stolpci.png')

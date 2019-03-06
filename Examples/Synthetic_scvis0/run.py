import sys
sys.path.append("../..")
import scimpute
import Orange.data
from Orange.data import Table
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors



# Izracunaj vse potrebno
data_gen = scimpute.generate()
dat, mas, zero = scimpute.zero_inflate(data_gen)
sc = scimpute.ScImpute(dat)
res = sc.scvis()
cor, data = sc.compare_embedded(res)
print(cor)

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

x = data[0]
y = data[1]

fig, axs = plt.subplots(2, 1)

axs[0].hist(x)
axs[0].set_title('Korelacija na učnih podatkih')
axs[1].hist(y)
axs[1].set_title('Korelacija na testnih podatkih')
fig.tight_layout()
fig.savefig('vrstice_stolpci.png')

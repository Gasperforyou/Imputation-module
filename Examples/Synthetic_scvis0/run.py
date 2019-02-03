import scimpute
import Orange.data
from Orange.data import Table
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors



# Izracunaj vse potrebno
data_gen = scimpute.generate()
dat, mas = scimpute.zero_inflate(data_gen)
sc = scimpute.ScImpute(dat)
res = sc.scvis()
cor, data = sc.compare(res[1][np.arange(res[0].shape[0]), :])
print(cor)

# Plotaj vse potrebno
# Primerjava bioloskih podatkov z imputiranimi vrednostmi
fig, (ax0, ax1) = plt.subplots(2, 1)
c = ax0.pcolormesh(res[0], cmap=plt.get_cmap("binary"))
fig.colorbar(c, ax=ax0)
ax0.set_title('Latentni prostor iz treniranja modela')

c = ax1.pcolormesh(res[1], cmap=plt.get_cmap("binary"))
ax1.set_title('Latentni prostor iz dodatnih podatkov')
fig.colorbar(c, ax=ax1)

fig.tight_layout()
fig.savefig('Latent1_latent2.png')


# Histogrami korelacij.

x = data[0]
y = data[1]

fig, axs = plt.subplots(2, 1)

axs[0].hist(x)
axs[0].set_title('Korelacija po vrsticah')
axs[1].hist(y)
axs[1].set_title('Korelacija po stolpcih')
fig.tight_layout()
fig.savefig('vrstice_stolpci.png')

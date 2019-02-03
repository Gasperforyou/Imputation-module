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
dat, mas, zero = scimpute.zero_inflate_bioloski(dataset.X)
sc = scimpute.ScImpute(dat)
res = sc.WMean_chisquared()
cor, data = sc.compare(dataset.X, mas, zero)
print(cor)
# Plotaj vse potrebno
# Primerjava bioloskih podatkov z imputiranimi vrednostmi
fig, (ax0, ax1) = plt.subplots(2, 1)
c = ax0.pcolormesh(dataset.X, norm=colors.LogNorm(vmin=np.amin(dataset.X)+1, vmax=np.amax(dataset.X)), cmap=plt.get_cmap("binary"))
fig.colorbar(c, ax=ax0)
ax0.set_title('Biološki podatki')

c = ax1.pcolormesh(res,norm=colors.LogNorm(vmin=np.amin(res)+1, vmax=np.amax(res)), cmap=plt.get_cmap("binary"))
ax1.set_title('Imputirani biološki podatki')
fig.colorbar(c, ax=ax1)

fig.tight_layout()
fig.savefig('biological_and_imputed.png')


# plotanje razlike
fig, (ax0) = plt.subplots()
c = ax0.pcolormesh(dataset.X-res, norm=colors.LogNorm(vmin=np.amin(dataset.X)+1, vmax=np.amax(dataset.X)), cmap=plt.get_cmap("binary"))
ax0.set_title('Razlika med bioloskimi podatki in imputacijo')
fig.colorbar(c, ax=ax0)

fig.tight_layout()
fig.savefig('razlika.png')

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

x = data[2]
y = data[3]

fig, axs = plt.subplots(2, 1)

axs[0].hist(x)
axs[0].set_title('Korelacija po maskiranih vrsticah')
axs[1].hist(y)
axs[1].set_title('Korelacija po maskiranih stolpcih')
fig.tight_layout()
fig.savefig('vrstice_stolpci_maskirano.png')

# np.savetxt("average.csv", res, delimiter=",")

# res = sc.average()
# np.savetxt("average.csv", res, delimiter=",")
#
# res = sc.median()
# np.savetxt("median.csv", res, delimiter=",")
#
# res = sc.WMean_chisquared()
# np.savetxt("WMean_chisquared.csv", res, delimiter=",")

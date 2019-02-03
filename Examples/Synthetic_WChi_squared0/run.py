import scimpute
import Orange.data
from Orange.data import Table
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors



# Izracunaj vse potrebno z uporabo modula
data_org = scimpute.generate()
dat, mas = scimpute.zero_inflate(data_org)
sc = scimpute.ScImpute(dat)
res = sc.WMean_chisquared()
cor, data = sc.compare(data_org, mas)

# Plotaj vse potrebno
# Primerjava bioloskih podatkov z imputiranimi vrednostmi
fig, (ax0, ax1) = plt.subplots(2, 1)
c = ax0.pcolormesh(data_org, norm=colors.LogNorm(vmin=np.amin(data_org)+1, vmax=np.amax(data_org)), cmap=plt.get_cmap("binary"))
fig.colorbar(c, ax=ax0)
ax0.set_title('Sintetični podatki')

c = ax1.pcolormesh(res,norm=colors.LogNorm(vmin=np.amin(res)+1, vmax=np.amax(res)), cmap=plt.get_cmap("binary"))
ax1.set_title('Imputirani sintetični podatki')
fig.colorbar(c, ax=ax1)

fig.tight_layout()
fig.savefig('synthetic_and_imputed.png')


# plotanje razlike
fig, (ax0) = plt.subplots()
c = ax0.pcolormesh(data_org-res, norm=colors.LogNorm(vmin=np.amin(data_org)+1, vmax=np.amax(data_org)), cmap=plt.get_cmap("binary"))
ax0.set_title('Razlika med sintetičnimi in imputiranimi podatki')
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

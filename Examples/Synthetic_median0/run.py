import sys
sys.path.append("../..")
import scimpute
import Orange.data
from Orange.data import Table
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors



# Izracunaj vse potrebno z uporabo modula
data_org = scimpute.generate()
dat, mas, zero = scimpute.zero_inflate(data_org)
sc = scimpute.ScImpute(dat)
res = sc.median()
cor, data = sc.compare(data_org, mas)
print(cor)
# Plotaj vse potrebno
# Primerjava bioloskih podatkov z imputiranimi vrednostmi
fig, (ax0, ax1, ax2) = plt.subplots(3, 1)
c = ax0.pcolormesh(data_org, norm=colors.LogNorm(vmin=np.amin(data_org)+1, vmax=np.amax(data_org)), cmap=plt.get_cmap("binary"))
fig.colorbar(c, ax=ax0)
ax0.set_title('Sintetični podatki')

c = ax1.pcolormesh(res,norm=colors.LogNorm(vmin=np.amin(res)+1, vmax=np.amax(res)), cmap=plt.get_cmap("binary"))
ax1.set_title('Imputirani sintetični podatki')
fig.colorbar(c, ax=ax1)

c = ax2.pcolormesh(data_org-res, norm=colors.LogNorm(vmin=np.amin(data_org)+1, vmax=np.amax(data_org)), cmap=plt.get_cmap("binary"))
ax2.set_title('Razlika med sintetičnimi in imputiranimi podatki')
fig.colorbar(c, ax=ax2)

fig.tight_layout()
fig.savefig('synthetic_and_imputed.png')

# histogrami matrik
fig, (ax0, ax1, ax2) = plt.subplots(3, 1)

ax0.hist(data_org.flatten())
ax0.set_title('Histogram sintetičnih podatkov')

ax1.hist(res.flatten())
ax1.set_title('Histogram imputiranih podatkov')

ax2.hist((data_org-res).flatten())
ax2.set_title('Histogram razlike med sintetičnimi in imputiranimi podatki')
fig.tight_layout()
fig.savefig('histogram_matrik.png')

# Histogrami korelacij.

x = data[0]
y = data[1]
z = data[2]
v = data[3]

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

# np.savetxt("average.csv", res, delimiter=",")

# res = sc.average()
# np.savetxt("average.csv", res, delimiter=",")
#
# res = sc.median()
# np.savetxt("median.csv", res, delimiter=",")
#
# res = sc.WMean_chisquared()
# np.savetxt("WMean_chisquared.csv", res, delimiter=",")

import scimpute
import Orange.data
from Orange.data import Table
import numpy as np
import matplotlib.pyplot as plt


# Load
filename = "../data/ccp_normCounts_mESCquartz.counts.cycle_genes.csv"
dataset = Orange.data.Table(filename)

mas, zero = scimpute.zero_inflate_bioloski(dataset.X)

file_inflated = np.genfromtxt('biological_inflated.csv', delimiter=',')

sc = scimpute.ScImpute(file_inflated)

res = sc.average()

cor = sc.compare(dataset.X, mas, zero)


fig, (ax0, ax1) = plt.subplots(2, 1)
c = ax0.pcolormesh(dataset.X)
fig.colorbar(c, ax=ax0)
ax0.set_title('default: no edges')

c = ax1.pcolormesh(res)
ax1.set_title('thick edges')
fig.colorbar(c, ax=ax1)

fig.tight_layout()
fig.savefig('biological_and_imputed.png')


# np.savetxt("average.csv", res, delimiter=",")

# res = sc.average()
# np.savetxt("average.csv", res, delimiter=",")
#
# res = sc.median()
# np.savetxt("median.csv", res, delimiter=",")
#
# res = sc.WMean_chisquared()
# np.savetxt("WMean_chisquared.csv", res, delimiter=",")

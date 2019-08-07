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
sc = scimpute.ScImpute(data_gen)
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

with open('your_file.txt', 'w') as f:
    for item in data[0]:
        f.write("%s\n" % item)
with open('your_file2.txt', 'w') as f:
    for item in data[1]:
        f.write("%s\n" % item)

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
axs[0].set_title('Korelacija večja od 0.7')
axs[1].hist(razlika2)
axs[1].set_title('Korelacija manjša od 0.3')
fig.tight_layout()
fig.savefig('vrstice_stolpci.png')

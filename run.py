import Orange.data
from Orange.data import Table
from pandas import DataFrame, read_csv


import owimpute

# scVi : [1000, 1e-4, True, True, 0.8, 5]
# scvis : [0.0002, 10, 10, 0.001, 0, 1, 7]
# pCMF : [2, True, False, False, 8]

# Generate sinthetic data
# owimpute.generate()

# Choose original filename

filename = "./data/ccp_normCountsBuettnerEtAl.counts.cycle_genes.csv"
# filename = "sinthetic_original.csv"

# Zeroinflate data
# zr, zeros = owimpute.zero_inflate()
zr, zeros = owimpute.zero_inflate_bioloski(filename)

# Start app
from AnyQt.QtWidgets import QApplication
app = QApplication([])

# Create object, set data and choose the method
ow = owimpute.Impute(filename, zr, zeros)

# dataset = Orange.data.Table("sinthetic_original_inflated.csv")
dataset = Orange.data.Table("biological_inflated.csv")

ow.set_data(dataset)
results = ow.selection("scvis", [0.0002, 10, 10, 0.001, 0, 1, 7])
# originalno matriko
# original = Table(filename)
# original = read_csv(filename, sep=",", header= None)
# original = Table.from_numpy(Orange.data.Domain.from_numpy(original.values), original.values)

# res = ow.compare(original)
res = ow.compare()
print(res)

import sys
import numpy
from PyQt5 import QtCore, QtGui
import Orange.data
from Orange.data import Table

from subprocess import call

import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


from scvi.dataset import CsvDataset, BreastCancerDataset, BrainSmallDataset, CortexDataset

from scvi.models import *
from scvi.inference import UnsupervisedTrainer


from Orange.widgets.widget import OWWidget, Input, Output, settings
from Orange.widgets import gui

from Orange.preprocess import Impute, Average


class Impute(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "Impute"
    icon = "icons/mywidget.svg"
    priority = 1000


    class Inputs:
        data = Input("Data", Orange.data.Table)

    class Outputs:
        sample = Output("Imputed Data", Orange.data.Table)

    want_main_area = False

    methods =[]
    imputation_method = []
    proportion = settings.Setting(50)
    Commit = "Impute"
    train = []
    estimate = []
    compute = []
    path = ""
    functions = {0: "flow", 1: "self.average()", 2: "self.median()", 3:"self.WMean_chisquared()", 4:"self.scvi()"}
    gene = ""

    def __init__(self, pat):
        super().__init__()
        self.path = pat
        self.listbox = gui.listBox(self.controlArea, self, "imputation_method", "methods" , box = "Imputation method")
        self.methods = ["Flow through", "Average", "Median", "WMean_chisquared", "scVi"]
        self.imputation_method = gui.ControlledList([0], self.listbox)
        gui.button(self.controlArea, self, "Impute", callback=self.comm)


    @Inputs.data
    def set_data(self, dataset):
        if dataset is not None:
            self.dataset = dataset
        else:
            self.dataset = None
        #self.commit()

    def comm(self):
        self.selection()
        self.commit()

    def selection(self):
        if self.dataset is None:
            return
        if(self.imputation_method[0]<=3):
            self.impute()
        else:
            func = self.functions[self.imputation_method[0]]
            eval(func)
            print("Done!")


    def commit(self):
        if self.dataset is None:
            return
        self.Outputs.sample.send(self.dataset)
        Orange.data.Table.save(self.dataset, "output.csv")
        return

    def impute(self):
        func = self.functions[self.imputation_method[0]]
        if(func=="flow"):
            return
        self.compute = numpy.copy(self.dataset.X)
        for dom in range(self.compute.shape[1]):
            self.estimate = numpy.where(self.compute[:,dom] == 0)
            self.train = self.compute[numpy.where(self.compute[:,dom] != 0)]
            if(self.train.shape[0]==0):
                continue
            self.gene = dom
            eval(func)
        print("Done!")
        return


    def average(self):
        avg = numpy.mean(self.train[:,self.gene])
        for d in self.estimate[0]:
            self.dataset[d][self.gene] = avg

    def median(self):
        med = numpy.median(self.train[:,self.gene])
        for d in self.estimate[0]:
            self.dataset[d][self.gene] = med

    def WMean_chisquared(self):
        avg, sum_weights = numpy.average(self.train[:,self.gene], weights=numpy.random.chisquare(3, self.train.shape[0]), returned=True)
        for d in self.estimate[0]:
            self.dataset[d][self.gene] = avg

    def scvi(self):
        #datasetBreastCancer = BreastCancerDataset()
        #datasetBrainSmall = BrainSmallDataset()
        dataset = CortexDataset()

        n_epochs=400
        lr=1e-3
        use_batches=False
        use_cuda=True

        vae = VAE(dataset.nb_genes, n_batch=dataset.n_batches * use_batches)
        trainer = UnsupervisedTrainer(vae, dataset ,train_size=0.75, use_cuda=use_cuda ,frequency=5)
        trainer.train(n_epochs=n_epochs, lr=lr)


        ll_train_set = trainer.history["ll_train_set"]
        ll_test_set = trainer.history["ll_test_set"]
        x = numpy.linspace(0,500,(len(ll_train_set)))
        plt.plot(x, ll_train_set)
        plt.plot(x, ll_test_set)
        plt.ylim(1150,1600)
        plt.show()

        trainer.train_set.show_t_sne(n_samples=400, color_by='labels')
        return

    def  scvis(self):
        return





def main(argv=None):
    from AnyQt.QtWidgets import QApplication
    # PyQt changes argv list in-place
    app = QApplication(list(argv) if argv else [])
    argv = app.arguments()
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "./data/_sc_aml-1k.csv"

    ow = Impute(filename)
    ow.show()
    ow.raise_()

    dataset = Orange.data.Table(filename)

    ow.set_data(dataset)
    ow.handleNewSignals()
    app.exec_()
    ow.set_data(None)
    ow.handleNewSignals()
    ow.onDeleteWidget()
    call(["python", "owtable.py"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

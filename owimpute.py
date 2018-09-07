import sys
import numpy
from PyQt5 import QtCore, QtGui
import Orange.data
from Orange.data import Table
from subprocess import call
import os
from sklearn.manifold import TSNE
# import matplotlib.pyplot
# import matplotlib.pyplot as p
from scvi.dataset import CsvDataset, BreastCancerDataset, BrainSmallDataset, CortexDataset
from scvi.models import *

from Orange.widgets.widget import OWWidget, Input, Output, settings
from Orange.widgets import gui
from Orange.preprocess import Impute, Average
import rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
from rpy2 import robjects as ro
import csv
from pandas import DataFrame, read_csv
from rpy2.robjects import numpy2ri





os.environ['TF_CPP_MIN_LOG_LEVEL']='0'


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
    functions = {0: "flow", 1: "self.average()", 2: "self.median()", 3:"self.WMean_chisquared()", 4:"self.scvi()", 5:"self.scvis()", 6:"self.pCMF()"}
    gene = ""

    def __init__(self, pat):
        super().__init__()
        self.path = pat
        self.listbox = gui.listBox(self.controlArea, self, "imputation_method", "methods" , box = "Imputation method")
        self.methods = ["Flow through", "Average", "Median", "WMean_chisquared", "scVi", "scvis", "pCMF"]
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

    def prepare_data(self):
        # local_csv_dataset = CsvDataset("tmp.csv", save_path="./")
        # a = numpy.array([14, 21, 13, 56, 12])
        # labels = ["ena", "dva", "tri", "štiri", "pet"]
        # print(self.dataset.domain, self.dataset.W, self.dataset.metas)
        if self.functions[self.imputation_method[0]] == "self.scvi()":
            df2 = DataFrame(self.dataset.X,   columns=self.dataset.domain.variables).T.to_csv("tmp.csv")
        if self.functions[self.imputation_method[0]] == "self.scvis()":
            df2 = DataFrame(self.dataset.X,   columns=self.dataset.domain.variables).to_csv("tmp.csv", sep='\t')
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
        from scvi.inference import UnsupervisedTrainer
        #datasetBreastCancer = BreastCancerDataset()
        #datasetBrainSmall = BrainSmallDataset()
        #dataset = CortexDataset()
        self.prepare_data()
        dataset = CsvDataset("tmp.csv", save_path="./")

        n_epochs=400
        lr=1e-3
        use_batches=False
        use_cuda=True

        vae = VAE(dataset.nb_genes, n_batch=dataset.n_batches * use_batches)
        trainer = UnsupervisedTrainer(vae, dataset ,train_size=0.75, use_cuda=use_cuda ,frequency=5)
        trainer.train(n_epochs=n_epochs, lr=lr)

        indices = trainer.train_set.indices
        self.dataset.X = trainer.train_set.sequential().imputation()
        return

    def scvis(self):
        self.prepare_data()
        call(["python", ".\scvis-dev\scvis", "train", "--data_matrix_file", ".\\tmp.csv", "--config_file", ".\\scvis-dev\\model_config.yaml" , "--show_plot", "--verbose", "--verbose_interval", "50", "--out_dir", ".\\output"])
        data = read_csv(".\\output\\traned_data.tsv", sep="\t")
        data = data.drop(data.columns[0], axis=1)
        self.dataset.X = data.values
        return

    def pCMF(self):
        utils = importr("utils")

        # install packages
        # ro.r('''   install.packages("devtools")
        #             devtools::install_git("https://gitlab.inria.fr/gdurif/pCMF", subdir="pkg")
        # ''')

        d = {'print.me': 'print_dot_me', 'print_me': 'print_uscore_me'}
        try:
            pCMF = importr('pCMF', robject_translations = d, lib_loc = "C:/Users/Gasper/Documents/R/win-library/3.4")
        except:
            print("error")

        try:
            labeling = importr('labeling', robject_translations = d, lib_loc = "C:/Users/Gasper/Documents/R/win-library/3.4")
        except:
            print("error")
        numpy2ri.activate()
        nr,nc = self.dataset.X.shape
        Br = ro.r.matrix(self.dataset.X, nrow=nr, ncol=nc)
        ro.r.assign("dataset", Br)

        ro.r('''

        # n <- 9
        # p <- 500
        # K <- 10
        # factorU <- generate_factor_matrix(n, K, ngroup=3, average_signal=60,
        #                                   group_separation=0.8,
        #                                   distribution="gamma",
        #                                   shuffle_feature=TRUE)
        # factorV <- generate_factor_matrix(p, K, ngroup=2, average_signal=60,
        #                                   group_separation=0.8,
        #                                   distribution="gamma",
        #                                   shuffle_feature=TRUE,
        #                                   prop_noise_feature=0.4,
        #                                   noise_level=0.5)
        # U <- factorU$factor_matrix
        # V <- factorV$factor_matrix
        # count_data <- generate_count_matrix(n, p, K, U, V,
        #                                     ZI=TRUE, prob1=rep(0.5,p))
        # X <- count_data$X
        X <- dataset
        print(dim(X))
        # matrix_heatmap(X)

        {res1 <- pCMF(X, K=2, verbose=TRUE, zero_inflation = TRUE,
             sparsity = TRUE, ncores=8);}

        # ## estimated probabilities
        # matrix_heatmap(res1$sparse_param$prob_S)
        # ## corresponding indicator (prob_S > threshold, where threshold = 0.5)
        # matrix_heatmap(res1$sparse_param$S)
        # ## rerun with genes that contributes
        # res2 <- pCMF(X[,res1$sparse_param$prior_prob_S>0],
        #              K=2, verbose=FALSE, zero_inflation = TRUE,
        #              sparsity = FALSE, ncores=8)
        #
        # #hatU <- getU(res)
        # #hatV <- getV(res)
        #
        # graphU(res2, axes=c(1,2), labels=factorU$feature_label)
        ''')
        numpy2ri.deactivate()
        return





def main(argv=None):

    from AnyQt.QtWidgets import QApplication
    # PyQt changes argv list in-place
    app = QApplication(list(argv) if argv else [])
    argv = app.arguments()
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "./data/ccp_normCountsBuettnerEtAl.counts.cycle_genes.csv"

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

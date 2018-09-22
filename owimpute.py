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
import inspect
import scipy

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


    def __init__(self, pat, zr, z = None):
        super().__init__()
        self.zeros = z
        self.test= numpy.array(zr)
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

        if self.zeros is not None:
            self.dataset.X = self.dataset.X*self.zeros

        self.Outputs.sample.send(self.dataset)
        self.dataset.save("output.csv")

        # original = Table("./data/ccp_normCountsBuettnerEtAl.counts.cycle_genes.csv")
        original = read_csv(".\\original.csv", sep=",", header= None)
        original = Table.from_numpy(Orange.data.Domain.from_numpy(original.values), original.values)

        self.spearman(original)
        print(self.spear[:, 0].mean())
        Table.from_numpy(Orange.data.Domain.from_numpy(self.spear), self.spear).save("correlation_c.csv")
        self.spearmanT(original)
        print(self.spear[:, 0].mean())
        Table.from_numpy(Orange.data.Domain.from_numpy(self.spear), self.spear).save("correlation_g.csv")
        self.spearmanM(original)
        print(self.spear[:, 0].mean())
        Table.from_numpy(Orange.data.Domain.from_numpy(self.spear), self.spear).save("correlation_m.csv")
        self.spearmanMT(original)
        print(self.spear[:, 0].mean())
        Table.from_numpy(Orange.data.Domain.from_numpy(self.spear), self.spear).save("correlation_n.csv")
        p = scipy.stats.spearmanr(original.X, self.dataset.X, axis = None)
        print(p)
        return

    def spearman(self, original):
        vec = numpy.zeros((original.X.shape[0], 2))
        for a in range(original.X.shape[0]):
            vec[a, 0] = scipy.stats.spearmanr(original.X[a,:], self.dataset.X[a,:])[0]
        self.spear = vec
        return

    def spearmanT(self, original):
        vec = numpy.zeros((original.X.shape[1], 2))
        for a in range(original.X.shape[1]):
            vec[a, 0] = scipy.stats.spearmanr(original.X[:,a], self.dataset.X[:,a])[0]
        self.spear = vec
        return

    def spearmanM(self, original):
        x = numpy.ma.masked_array(original.X, mask = 1-self.test)
        y = numpy.ma.masked_array(self.dataset.X, mask = 1-self.test)
        vec = numpy.zeros((x.shape[0], 2))
        for a in range(x.shape[0]):
            vec[a, 0] = scipy.stats.mstats.spearmanr(x[a,:], y[a,:])[0]
        self.spear = vec
        return

    def spearmanMT(self, original):
        x = numpy.ma.masked_array(original.X, mask = 1-self.test)
        y = numpy.ma.masked_array(self.dataset.X, mask = 1-self.test)
        vec = numpy.zeros((x.shape[1], 2))
        for a in range(x.shape[1]):
            vec[a, 0] = scipy.stats.mstats.spearmanr(x[:,a], y[:,a])[0]
        self.spear = vec
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

        # if self.functions[self.imputation_method[0]] == "self.scvi()":
        #     df2 = DataFrame(self.dataset.X.astype(int),   columns=self.dataset.domain.variables[:-1]).T.to_csv("tmp.csv")
        # if self.functions[self.imputation_method[0]] == "self.scvis()":
        #     df2 = DataFrame(self.dataset.X.astype(int),   columns=self.dataset.domain.variables[:-1]).to_csv("tmp.csv", sep='\t')

        if self.functions[self.imputation_method[0]] == "self.pCMF()":
            df2 = DataFrame(self.dataset.X).to_csv("tmp.csv", sep=',', header=None, index=False)
        if self.functions[self.imputation_method[0]] == "self.scvi()":
            df2 = DataFrame(self.dataset.X,  columns=self.dataset.domain.variables).T.to_csv("tmp.csv")
        if self.functions[self.imputation_method[0]] == "self.scvis()":
            df2 = DataFrame(self.dataset.X, columns=self.dataset.domain.variables).to_csv("tmp.csv", sep='\t')

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
        self.prepare_data()
        data = CsvDataset("tmp.csv", save_path="./", sep=',')

        n_epochs = 500
        lr=1e-4
        use_batches=True
        use_cuda=True

        vae = VAE(data.nb_genes, n_batch=data.n_batches * use_batches)
        trainer = UnsupervisedTrainer(vae, data ,train_size=0.8, use_cuda=use_cuda ,frequency=5)
        trainer.train(n_epochs=n_epochs, lr=lr)

        indices1 = trainer.train_set.indices
        indices2 = trainer.test_set.indices
        datac = numpy.append(trainer.train_set.sequential().imputation(), trainer.test_set.sequential().imputation() , axis=0)
        ind = numpy.append(indices1, indices2)
        ind = numpy.argsort(ind)
        self.dataset.X = datac[ind, :].astype(float)

        return

    def scvis(self):
        self.prepare_data()
        call(["python", ".\scvis-dev\scvis", "train", "--data_matrix_file", ".\\tmp.csv", "--config_file", ".\\scvis-dev\\model_config.yaml" , "--show_plot", "--verbose", "--verbose_interval", "50", "--out_dir", ".\\output"])
        data = read_csv(".\\output\\traned_data.tsv", sep="\t")
        data = data.drop(data.columns[0], axis=1)
        self.dataset.X = data.values
        return

    def pCMF(self):
        self.prepare_data()
        utils = importr("utils")
        print(self.dataset.X.shape)

        # install packages
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

        print(ro.r('''
        dat <- read.csv(file='tmp.csv', header = FALSE)
        dat <- as.matrix(dat)

        zeros <- as.vector(which(!(colSums(dat != 0) > 0)))
        zeros_mat <- dat[,zeros]

        non_zeros <- as.vector(which((colSums(dat != 0) > 0)))
        dat <- dat[, non_zeros]


        res1 <- pCMF(dat, K=2, verbose=TRUE, zero_inflation = FALSE,
             sparsity = FALSE, ncores=8)


        hatU <- getU(res1)
        hatV <- getV(res1)


        out <- hatU %*% t(hatV)
        out<- unname(cbind(out, zeros_mat))
        out <- out[, order(c(non_zeros, zeros))]
        '''))

        self.dataset.X = numpy.asarray(ro.r.out)
        return

def generate():
    utils = importr("utils")
    d = {'print.me': 'print_dot_me', 'print_me': 'print_uscore_me'}
    try:
        pCMF = importr('pCMF', robject_translations = d, lib_loc = "C:/Users/Gasper/Documents/R/win-library/3.4")
    except:
        print("error")

    try:
        labeling = importr('labeling', robject_translations = d, lib_loc = "C:/Users/Gasper/Documents/R/win-library/3.4")
    except:
        print("error")
    utils = importr("utils")
    ro.r('''
        n <- 100
        p <- 300
        K <- 3
        factorU <- generate_factor_matrix(n, K, ngroup=3, average_signal=60,
                                          group_separation=0.8,
                                          distribution="gamma",
                                          shuffle_feature=TRUE)
        factorV <- generate_factor_matrix(p, K, ngroup=2, average_signal=60,
                                          group_separation=0.8,
                                          distribution="gamma",
                                          shuffle_feature=TRUE,
                                          prop_noise_feature=0.4,
                                          noise_level=0.5)
        U <- factorU$factor_matrix
        V <- factorV$factor_matrix
        count_data <- generate_count_matrix(n, p, K, U, V,
                                            ZI=FALSE, prob1=rep(0.5,p))
        X <- count_data$X
        write.table(X, file = "original.csv",row.names=FALSE, na="",col.names=FALSE, sep=",")
    ''')
    return

def zero_inflate():
    return ro.r('''
    dat <- read.csv(file='original.csv', header = FALSE)
    dat <- as.matrix(dat)
    m <- matrix(sample(0:1,nrow(dat)*ncol(dat), replace=TRUE, prob=c(1,2)),nrow(dat),ncol(dat))
    dat <- dat*m
    write.table(dat, file = "pCMF.csv",row.names=FALSE, na="",col.names=FALSE, sep=",")
    zr <- (1-m)
    '''), None

def zero_inflate_bioloski():
    utils = importr("utils")
    numpy2ri.activate()

    tab = Table("./data/ccp_normCountsBuettnerEtAl.counts.cycle_genes.csv")

    nr,nc = tab.X.shape
    Br = ro.r.matrix(tab.X, nrow=nr, ncol=nc)
    ro.r.assign("tab", Br)

    zr = ro.r('''
    dat <- tab
    fo <- as.matrix((dat != 0))
    mode(fo) <- "integer"
    m <- matrix(sample(0:1,nrow(dat)*ncol(dat), replace=TRUE, prob=c(1,2)),nrow(dat),ncol(dat))
    dat <- dat*m
    write.table(dat, file = "pCMF.csv",row.names=FALSE, na="",col.names=FALSE, sep=",")
    zr <- fo*(1-m)
    ''')

    zeros = ro.r('fo')

    numpy2ri.deactivate()
    return zr, zeros


def main(argv=None):
    generate()
    zr, zeros = zero_inflate()

    # zr, zeros = zero_inflate_bioloski()
    from AnyQt.QtWidgets import QApplication
    # PyQt changes argv list in-place
    app = QApplication(list(argv) if argv else [])
    argv = app.arguments()
    if len(argv) > 1:
        filename = argv[1]
    else:
        filename = "./pCMF.csv"
        # filename = "./data/_sc_aml-1k.csv"


    ow = Impute(filename, zr, zeros)
    ow.show()
    ow.raise_()
    dataset = Orange.data.Table(filename)
    ow.set_data(dataset)
    ow.handleNewSignals()
    app.exec_()
    ow.set_data(None)
    ow.handleNewSignals()
    ow.onDeleteWidget()
    # call(["python", "owtable.py"])
    # call(["python", "owscaterplot.py"])
    # call(["python", "owheatmap.py"])
    # call(["python", "owdistribution.py"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

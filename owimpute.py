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
import yaml

os.environ['TF_CPP_MIN_LOG_LEVEL']='0'



class Impute(OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "Impute"
    icon = "icons/mywidget.svg"


    class Inputs:
        data = Input("Data", Orange.data.Table)

    class Outputs:
        sample = Output("Imputed Data", Orange.data.Table)

    want_main_area = False
    methods =[]
    imputation_method = []
    settings = [[],[],[],[],
    [1500, 1e-4, True, True, 0.8, 5],
    [0.0002, 10, 10, 0.001, 0, 1, 7],
    [2, True, False, False, 8]]
    Commit = "Impute"
    train = []
    estimate = []
    compute = []
    path = ""
    functions = {0: "flow", 1: "self.average()", 2: "self.median()", 3:"self.WMean_chisquared()", 4:"self.scvi()", 5:"self.scvis()", 6:"self.pCMF()"}
    gene = ""
    results = []


    def __init__(self, pat, zr, z = None):
        super().__init__()
        self.zeros = z
        self.test= numpy.array(zr)
        self.path = pat
        self.listbox = gui.listBox(self.controlArea, self, "imputation_method",
        "methods" , box = "Imputation method")
        self.methods = ["Flow through", "Average", "Median", "WMean_chisquared",
        "scVi", "scvis", "pCMF"]
        self.imputation_method = gui.ControlledList([0], self.listbox)
        gui.button(self.controlArea, self, "Impute", callback=self.selection)


    @Inputs.data
    def set_data(self, dataset):
        if dataset is not None:
            self.dataset = dataset
        else:
            self.dataset = None


    # Metode za izbiranje načina zaganjanja (kot modul ali gui widget)
    def selection(self, metoda = None, arg = None):

        if not metoda:
            metoda = self.imputation_method[0]
        else:
            metoda = self.methods.index(metoda)
            self.imputation_method = [metoda]

        if self.dataset is None:
            return

        if arg:
            self.settings[metoda] = arg

        if(metoda<=3):
            self.impute()
            print("Done!")

        else:
            func = self.functions[metoda]
            eval(func)
            print("Done!")

        return self.dataset
    # Metode za racunanje spearmanovih koeicientov
    def compare(self, original = None):
        if self.dataset is None:
            return

        self.Outputs.sample.send(self.dataset)
        self.dataset.save("output.csv")

        # if method not scvis
        if self.imputation_method[0] != 5:

            if self.zeros is not None:
                self.dataset.X = self.dataset.X*self.zeros

            # razlika = ((original.X + numpy.amin(original.X))/numpy.amax(original.X)) - ((self.dataset.X + numpy.amin(self.dataset.X))/numpy.amax(self.dataset.X))
            # Table.from_numpy(Orange.data.Domain.from_numpy(razlika), razlika).save("difference.csv")

            # odstrani ničelne stolpce
            res = []
            zero_columns = ~numpy.all(self.dataset.X == 0, axis=1)
            self.dataset.X= self.dataset.X[zero_columns]
            original = original[zero_columns]
            print(numpy.isnan(self.dataset.X))

            # spear po vrsticah
            self.spearman(original)
            self.spear = numpy.delete(self.spear, numpy.where(numpy.isnan(self.spear))[0], 0)
            res.append(numpy.absolute(self.spear[:, 0]).mean())
            Table.from_numpy(Orange.data.Domain.from_numpy(self.spear), self.spear).save("correlation_c.csv")
            # spear po stolpcih
            self.spearmanT(original)
            self.spear = numpy.delete(self.spear, numpy.where(numpy.isnan(self.spear))[0], 0)
            res.append(numpy.absolute(self.spear[:, 0]).mean())
            Table.from_numpy(Orange.data.Domain.from_numpy(self.spear), self.spear).save("correlation_g.csv")

            # Maskiran spear po vrsticah
            self.spearmanM(original)
            self.spear = numpy.delete(self.spear, numpy.where(numpy.isnan(self.spear))[0], 0)
            res.append(numpy.absolute(self.spear[:, 0]).mean())
            Table.from_numpy(Orange.data.Domain.from_numpy(self.spear), self.spear).save("correlation_m.csv")

            # Maskiran spear po stolpcih
            self.spearmanMT(original)
            self.spear = numpy.delete(self.spear, numpy.where(numpy.isnan(self.spear))[0], 0)
            res.append(numpy.absolute(self.spear[:, 0]).mean())
            Table.from_numpy(Orange.data.Domain.from_numpy(self.spear), self.spear).save("correlation_n.csv")

            p = scipy.stats.spearmanr(original.X, self.dataset.X, axis = None)
            res.append(p.correlation)

            return res
        else:
            return self.results

    # Izracunaj speraman koeficient po vrsticah
    def spearman(self, original):
        vec = numpy.zeros((original.X.shape[0], 2))
        for a in range(original.X.shape[0]):
            vec[a, 0] = scipy.stats.spearmanr(original.X[a,:], self.dataset.X[a,:])[0]
        self.spear = vec
        return
    # Izracunaj speraman koeficient po stolpcih
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
            # spearman rabi vsaj 3 nemaskirane vrednosti.
            # Preskoci vrstice s manj kot tremi vrednostmi.
            if(x[:,a].count()<3):
                vec[a, 0] = numpy.nan
                continue
            vec[a, 0] = scipy.stats.mstats.spearmanr(x[a,:], y[a,:])[0]
        self.spear = vec
        return

    def spearmanMT(self, original):
        x = numpy.ma.masked_array(original.X, mask = 1-self.test)
        y = numpy.ma.masked_array(self.dataset.X, mask = 1-self.test)
        vec = numpy.zeros((x.shape[1], 2))
        # Če je eden izmed vektorjev enak pri vseh vrednostih potem je korelacija irelavantna
        y[0] = y[0]+1e-6
        for a in range(x.shape[1]):
            # stolpec mora imeti vsaj 3 vrednosti za spearman funcijo
            # preskoci stolpce z manj kot 3 vrednostmi
            if(x[:,a].count()<3):
                vec[a, 0] = numpy.nan
                continue
            vec[a, 0] = scipy.stats.mstats.spearmanr(x[:,a], y[:,a])[0]
        self.spear = vec
        return

    # Metoda za členitev po stolpcih za prve tri metode imputacije
    # (mediana, povprecje, uteženo povprecje)
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

    # Metoda za pretvarjanje podatkov v ustrezen format
    def prepare_data(self):

        # if self.functions[self.imputation_method[0]] == "self.scvi()":
        #     df2 = DataFrame(self.dataset.X.astype(int),   columns=self.dataset.domain.variables[:-1]).T.to_csv("tmp.csv")
        # if self.functions[self.imputation_method[0]] == "self.scvis":
        #     idx = numpy.random.randint(self.dataset.X.shape[0], size=int(self.dataset.X.shape[0]*0.6))
        #     df2 = DataFrame(self.dataset.X[idx, :], columns=self.dataset.domain.variables[:-1]).to_csv("tmp_train_scvis.csv", sep='\t')
        #     df2 = DataFrame(self.dataset.X, columns=self.dataset.domain.variables[:-1]).to_csv("tmp.csv", sep='\t')

        if self.functions[self.imputation_method[0]] == "self.pCMF()":
            df2 = DataFrame(self.dataset.X).to_csv("tmp.csv", sep=',', header=None, index=False)
        if self.functions[self.imputation_method[0]] == "self.scvi()":
            df2 = DataFrame(self.dataset.X, columns=self.dataset.domain.variables).T.to_csv("tmp.csv")
        if self.functions[self.imputation_method[0]] == "self.scvis()":
            idx = numpy.random.randint(self.dataset.X.shape[0], size=int(self.dataset.X.shape[0]*0.75))
            df2 = DataFrame(self.dataset.X[idx, :], columns=self.dataset.domain.variables).to_csv("tmp_train_scvis.csv", sep='\t')
            df2 = DataFrame(self.dataset.X, columns=self.dataset.domain.variables).to_csv("tmp.csv", sep='\t')

        return

    # Metode za imputacijo
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


        # Set settings from self.settings
        # n_epochs = 1500
        # lr=1e-4
        # use_batches=True
        # use_cuda=True
        # tr_size = 0.8
        # freq = 5

        n_epochs = self.settings[self.imputation_method[0]][0]
        lr=self.settings[self.imputation_method[0]][1]
        use_batches=self.settings[self.imputation_method[0]][2]
        use_cuda=self.settings[self.imputation_method[0]][3]
        tr_size = self.settings[self.imputation_method[0]][4]
        freq = self.settings[self.imputation_method[0]][5]

        # settings and run
        vae = VAE(data.nb_genes, n_batch=data.n_batches * use_batches)
        trainer = UnsupervisedTrainer(vae, data ,train_size=tr_size, use_cuda=use_cuda ,frequency=freq)
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


        # Create configuration file using self.settings
        metoda = self.imputation_method[0]
        data1 = dict(
        hyperparameter = dict(
            optimization = dict(
            method = "Adam",
            learning_rate = self.settings[metoda][0]
            ),

            batch_size = self.settings[metoda][1],
            max_epoch = self.settings[metoda][2],
            regularizer_l2 = self.settings[metoda][3],

            perplexity = self.settings[metoda][4],

            seed = self.settings[metoda][5]
            )
        )

        data2 = dict(
        architecture = dict(
                latent_dimension = self.settings[metoda][6],

                inference = dict(
                layer_size = [128, 64, 32],
                ),

                model = dict(
                layer_size = [32, 32, 32, 64, 128],
                ),
                activation = "ELU"
            )
        )

        with open('.\\scvis-dev\\model_config.yaml', 'w') as outfile:
            yaml.dump(data1, outfile)
            yaml.dump(data2, outfile)

        # call train option
        call(["python", ".\scvis-dev\scvis", "train", "--data_matrix_file", ".\\tmp_train_scvis.csv", "--config_file", ".\\scvis-dev\\model_config.yaml" , "--show_plot", "--verbose", "--verbose_interval", "50", "--out_dir", ".\\output"])
        data = read_csv(".\\output\\traned_data.tsv", sep="\t")
        data = data.drop(data.columns[0], axis=1)
        self.dataset.X = data.values

        # call map option
        call(["python", ".\scvis-dev\scvis", "map", "--data_matrix_file", ".\\tmp.csv", "--config_file", ".\\scvis-dev\\model_config.yaml", "--out_dir", ".\\output1", "--pretrained_model_file", ".\\output\\model\\traned_data.ckpt"])
        # read results
        data2 = read_csv(".\\output1\\mapped.tsv", sep="\t")
        data2 = data2.drop(data2.columns[0], axis=1)
        data2 = Table.from_numpy(Orange.data.Domain.from_numpy(data2.values[numpy.arange(data.values.shape[0]), :]), data2.values[numpy.arange(data.values.shape[0]), :])

        razlika = ((data.values + numpy.amin(data.values))/numpy.amax(data.values)) - ((data2.X + numpy.amin(data2.X))/numpy.amax(data2.X))

        razlika = data.values-data2.X
        Table.from_numpy(Orange.data.Domain.from_numpy(razlika), razlika).save("difference.csv")

        self.spearman(data2)
        self.spear = numpy.delete(self.spear, numpy.where(numpy.isnan(self.spear))[0], 0)
        self.results.append(numpy.absolute(self.spear[:, 0]).mean())
        Table.from_numpy(Orange.data.Domain.from_numpy(self.spear), self.spear).save("correlation_c.csv")

        self.spearmanT(data2)
        self.spear = numpy.delete(self.spear, numpy.where(numpy.isnan(self.spear))[0], 0)
        self.results.append(numpy.absolute(self.spear[:, 0]).mean())
        Table.from_numpy(Orange.data.Domain.from_numpy(self.spear), self.spear).save("correlation_g.csv")

        p = scipy.stats.spearmanr(data, data2, axis = None)
        self.results.append(p.correlation)


    def pCMF(self):
        self.prepare_data()
        utils = importr("utils")

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

        # Assign self.settings
        metoda = self.imputation_method[0]
        ro.r.assign("a", self.settings[metoda][0])
        ro.r.assign("b", self.settings[metoda][1])
        ro.r.assign("c", self.settings[metoda][2])
        ro.r.assign("d", self.settings[metoda][3])
        ro.r.assign("e", self.settings[metoda][4])

        # Execute R code
        ro.r('''
        dat <- read.csv(file='tmp.csv', header = FALSE)
        dat <- as.matrix(dat)

        zeros <- as.vector(which(!(colSums(dat != 0) > 0)))
        zeros_mat <- dat[,zeros]

        non_zeros <- as.vector(which((colSums(dat != 0) > 0)))
        dat <- dat[, non_zeros]


        res1 <- pCMF(dat, K=a, verbose=b, zero_inflation = c,
             sparsity = d, ncores=e)


        hatU <- getU(res1)
        hatV <- getV(res1)


        out <- hatU %*% t(hatV)
        out<- unname(cbind(out, zeros_mat))
        out <- out[, order(c(non_zeros, zeros))]
        ''')

        self.dataset.X = numpy.asarray(ro.r.out)
        return


# Metoda za generiranje podatkov
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
        write.table(X, file = "sinthetic_original.csv",row.names=FALSE, na="",col.names=FALSE, sep=",")
    ''')
    return

# Metoda za umetno brisanje podatkov
def zero_inflate():
    return ro.r('''
    dat <- read.csv(file='sinthetic_original.csv', header = FALSE)
    dat <- as.matrix(dat)
    m <- matrix(sample(0:1,nrow(dat)*ncol(dat), replace=TRUE, prob=c(1,2)),nrow(dat),ncol(dat))
    dat <- dat*m
    write.table(dat, file = "sinthetic_original_inflated.csv",row.names=FALSE, na="",col.names=FALSE, sep=",")
    zr <- (1-m)
    '''), None
# Metoda za za umetno brisanje podatkov na bioloskih podatkih
def zero_inflate_bioloski(filename):
    utils = importr("utils")
    numpy2ri.activate()

    tab = Table(filename)

    nr,nc = tab.X.shape
    Br = ro.r.matrix(tab.X, nrow=nr, ncol=nc)
    ro.r.assign("tab", Br)

    zr = ro.r('''
    dat <- tab
    fo <- as.matrix((dat != 0))
    mode(fo) <- "integer"
    m <- matrix(sample(0:1,nrow(dat)*ncol(dat), replace=TRUE, prob=c(1,3)),nrow(dat),ncol(dat))
    dat <- dat*m
    write.table(dat, file = "biological_inflated.csv",row.names=FALSE, na="",col.names=FALSE, sep=",")
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
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

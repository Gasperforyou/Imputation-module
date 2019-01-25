import numpy as np
from pandas import DataFrame, read_csv
from scvi.inference import UnsupervisedTrainer
from scvi.dataset import CsvDataset
from scvi.models import *
import tempfile
import rpy2
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
from rpy2 import robjects as ro
from rpy2.robjects import numpy2ri
import yaml
from subprocess import call
import scipy


class ScImpute:

    # Sprejmi numpy matriko s podatki
    def __init__(self, dataset):
        self.dataset = dataset
        self.output = np.copy(self.dataset)

    # Metode za racunanje spearmanovih koeicientov
    def compare(self, original = None, mask = None, zeros = None):

        if zeros is not None:
            self.output = self.output*zeros

        # difference write on disk
        razlika = ((original - np.amin(original))/np.amax((original - np.amin(original)))) - ((self.output - np.amin(self.output))/np.amax((self.output - np.amin(self.output))))
        np.savetxt("razlika.csv", razlika, delimiter=",")

        res = []
        # spear po vrsticah
        self.spearman(original)
        self.spear = np.delete(self.spear, np.where(np.isnan(self.spear))[0], 0)
        res.append(np.absolute(self.spear[:, 0]).mean())
        np.savetxt("vrstice.csv", self.spear, delimiter=",")
        # spear po stolpcih
        self.spearmanT(original)
        self.spear = np.delete(self.spear, np.where(np.isnan(self.spear))[0], 0)
        res.append(np.absolute(self.spear[:, 0]).mean())
        np.savetxt("stolpci.csv", self.spear, delimiter=",")
        

        if mask is not None:
            # Maskiran spear po vrsticah
            self.spearmanM(original, mask)
            self.spear = np.delete(self.spear, np.where(np.isnan(self.spear))[0], 0)
            res.append(np.absolute(self.spear[:, 0]).mean())
            np.savetxt("vrstice_maskirano.csv", self.spear, delimiter=",")

            # Maskiran spear po stolpcih
            self.spearmanMT(original, mask)
            self.spear = np.delete(self.spear, np.where(np.isnan(self.spear))[0], 0)
            res.append(np.absolute(self.spear[:, 0]).mean())
            np.savetxt("stolpci_maskirano.csv", self.spear, delimiter=",")

        p = scipy.stats.spearmanr(original, self.output, axis = None)
        res.append(p.correlation)
        print(res)

        return res

    # Izracunaj speraman koeficient po vrsticah
    def spearman(self, original):
        vec = np.zeros((original.shape[0], 2))
        for a in range(original.shape[0]):
            vec[a, 0] = scipy.stats.spearmanr(original[a,:], self.output[a,:])[0]
        self.spear = vec
        return
    # Izracunaj speraman koeficient po stolpcih
    def spearmanT(self, original):
        vec = np.zeros((original.shape[1], 2))
        for a in range(original.shape[1]):
            vec[a, 0] = scipy.stats.spearmanr(original[:,a], self.output[:,a])[0]
        self.spear = vec
        return
    # Izracunaj maskiran speramanov koeficient po vrsticah
    def spearmanM(self, original, mask):
        x = np.ma.masked_array(original, mask = 1-mask)
        y = np.ma.masked_array(self.output, mask = 1-mask)
        vec = np.zeros((x.shape[0], 2))
        for a in range(x.shape[0]):
            # spearman rabi vsaj 3 nemaskirane vrednosti.
            # Preskoci vrstice s manj kot tremi vrednostmi.
            if(x[:,a].count()<3):
                vec[a, 0] = np.nan
                continue
            vec[a, 0] = scipy.stats.mstats.spearmanr(x[a,:], y[a,:])[0]
        self.spear = vec

        return
    # Izracunaj maskiran speramanov koeficient po stolpcih
    def spearmanMT(self, original, mask):
        x = np.ma.masked_array(original, mask = 1-mask)
        y = np.ma.masked_array(self.output, mask = 1-mask)
        vec = np.zeros((x.shape[1], 2))
        # Če je eden izmed vektorjev enak pri vseh vrednostih potem je korelacija irelavantna
        y[0] = y[0]+1e-6
        for a in range(x.shape[1]):
            # stolpec mora imeti vsaj 3 vrednosti za spearman funcijo
            # preskoci stolpce z manj kot 3 vrednostmi
            if(x[:,a].count()<3):
                vec[a, 0] = np.nan
                continue
            vec[a, 0] = scipy.stats.mstats.spearmanr(x[:,a], y[:,a])[0]
        self.spear = vec
        return

    # Metode za imputacijo
    def average(self):

        compute = np.copy(self.dataset)
        # Za vsak gen razdeli tabelo na train in estimate
        for gene in range(compute.shape[1]):
            estimate = np.where(compute[:,gene] == 0)
            train = compute[np.where(compute[:,gene] != 0)]
            if(train.shape[0]==0):
                continue
            # Racunaj in shrani
            avg = np.mean(train[:,gene])
            for d in estimate[0]:
                self.output[d][gene] = avg
        return self.output

    def median(self):

        compute = np.copy(self.dataset)
        # Za vsak gen razdeli tabelo na train in estimate
        for gene in range(compute.shape[1]):
            estimate = np.where(compute[:,gene] == 0)
            train = compute[np.where(compute[:,gene] != 0)]
            if(train.shape[0]==0):
                continue
            # Racunaj in shrani
            med = np.median(train[:,gene])
            for d in estimate[0]:
                self.output[d][gene] = med
        return self.output

    def WMean_chisquared(self):

        compute = np.copy(self.dataset)
        # Za vsak gen razdeli tabelo na train in estimate
        for gene in range(compute.shape[1]):
            estimate = np.where(compute[:,gene] == 0)
            train = compute[np.where(compute[:,gene] != 0)]
            if(train.shape[0]==0):
                continue
            # Racunaj in shrani
            avg, sum_weights = np.average(train[:,gene],
            weights=np.random.chisquare(3, train.shape[0]),
            returned=True)

            for d in estimate[0]:
                self.output[d][gene] = avg
        return self.output

    def scvi(self, **kwargs):
        # Create temporary directory for data conversion
        data = None
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Convert into ScvDataset format
            df2 = DataFrame(self.dataset,
            columns=self.dataset.dtype.names).T.to_csv(tmpdirname+"\\tmp.csv")
            # import data
            data = CsvDataset("\\tmp.csv", save_path=tmpdirname, sep=',')


        # Set settings
        n_epochs = kwargs["n_epochs"] if "n_epochs" in kwargs else 1500
        lr= kwargs["lr"] if "lr" in kwargs else 1e-4
        use_batches=kwargs["use_batches"] if "use_batches" in kwargs else True
        use_cuda=kwargs["use_cuda"] if "use_cuda" in kwargs else True
        tr_size = kwargs["tr_size"] if "tr_size" in kwargs else 0.8
        freq = kwargs["freq"] if "freq" in kwargs else 5

        # settings and run
        vae = VAE(data.nb_genes, n_batch=data.n_batches * use_batches)
        trainer = UnsupervisedTrainer(vae, data ,train_size=tr_size, use_cuda=use_cuda ,frequency=freq)
        trainer.train(n_epochs=n_epochs, lr=lr)

        # reconstruct the outpu matrix
        indices1 = trainer.train_set.indices
        indices2 = trainer.test_set.indices
        datac = np.append(trainer.train_set.sequential().imputation(), trainer.test_set.sequential().imputation() , axis=0)
        ind = np.append(indices1, indices2)
        ind = np.argsort(ind)
        self.output = datac[ind, :].astype(float)

        return self.output

    def pCMF(self, **kwargs):
        with tempfile.TemporaryDirectory() as tmpdirname:
            df2 = DataFrame(self.dataset).to_csv(tmpdirname+"\\tmp.csv", sep=',', header=None, index=False)

            utils = importr("utils")

            ro.r.assign("tmpdirname", tmpdirname)

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

            # Assign settings
            ro.r.assign("a", kwargs["k"] if "k" in kwargs else 2)
            ro.r.assign("b", kwargs["verbose"] if "verbose" in kwargs else True)
            ro.r.assign("c", kwargs["zero_inflation"] if "zero_inflation" in kwargs else False)
            ro.r.assign("d", kwargs["sparsity"] if "sparsity" in kwargs else False)
            ro.r.assign("e", kwargs["ncores"] if "ncores" in kwargs else 8)

            # Execute R code
            ro.r('''
            path <- paste(tmpdirname, 'tmp.csv', sep = '/')
            dat <- read.csv(file=path, header = FALSE)
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

            self.output = np.asarray(ro.r.out)
        return self.output

    def scvis(self, **kwargs):
        with tempfile.TemporaryDirectory() as tmpdirname:
            idx = np.random.randint(self.dataset.shape[0], size=int(self.dataset.shape[0]*0.75))
            df2 = DataFrame(self.dataset[idx, :], columns=self.dataset.dtype.names).to_csv(tmpdirname+"/tmp_train_scvis.csv", sep='\t')
            df2 = DataFrame(self.dataset, columns=self.dataset.dtype.names).to_csv(tmpdirname+"/tmp.csv", sep='\t')


            # Create configuration file
            data1 = dict(
            hyperparameter = dict(
                optimization = dict(
                method = "Adam",
                learning_rate = kwargs["learning_rate"] if "learning_rate" in kwargs else 0.0002
                ),

                batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 10,
                max_epoch = kwargs["max_epoch"] if "max_epoch" in kwargs else 10,
                regularizer_l2 = kwargs["regularizer_l2"] if "regularizer_l2" in kwargs else 0.001,

                perplexity = kwargs["perplexity"] if "perplexity" in kwargs else 0,

                seed = kwargs["seed"] if "seed" in kwargs else 1
                )
            )

            data2 = dict(
            architecture = dict(
                    latent_dimension = kwargs["latent_dimension"] if "latent_dimension" in kwargs else 7,

                    inference = dict(
                    layer_size = [128, 64, 32],
                    ),

                    model = dict(
                    layer_size = [32, 32, 32, 64, 128],
                    ),
                    activation = "ELU"
                )
            )

            with open(tmpdirname+'\\model_config.yaml', 'w') as outfile:
                yaml.dump(data1, outfile)
                yaml.dump(data2, outfile)

            # call train option
            call(["python", "./scvis-dev/scvis", "train", "--data_matrix_file", tmpdirname+"\\tmp_train_scvis.csv", "--config_file", tmpdirname+"\\model_config.yaml" , "--show_plot", "--verbose", "--verbose_interval", "50", "--out_dir", tmpdirname+"\\output"])
            data = read_csv(tmpdirname+"\\output\\traned_data.tsv", sep="\t")
            data = data.drop(data.columns[0], axis=1)
            self.output = data.values

            # call map option
            call(["python", ".\scvis-dev\scvis", "map", "--data_matrix_file", tmpdirname+"\\tmp.csv", "--config_file", tmpdirname+"\\model_config.yaml", "--out_dir", tmpdirname+"\\output1", "--pretrained_model_file", tmpdirname+"\\output\\model\\traned_data.ckpt"])
            # read results
            data2 = read_csv(tmpdirname+"\\output1\\mapped.tsv", sep="\t")
            data2 = data2.drop(data2.columns[0], axis=1)
        return self.output

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
def zero_inflate_bioloski(file):
    utils = importr("utils")
    numpy2ri.activate()


    nr,nc = file.shape
    Br = ro.r.matrix(file, nrow=nr, ncol=nc)
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

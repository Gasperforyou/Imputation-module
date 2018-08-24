import sys
import numpy
from PyQt5 import QtCore, QtGui
import Orange.data
from Orange.data import Table
import scvi


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
    functions = {0: "True", 1: "self.average()"}
    gene = ""

    def __init__(self):
        super().__init__()

        self.listbox = gui.listBox(self.controlArea, self, "imputation_method", "methods" , box = "Imputation method")
        self.methods = ["Flow through", "Average"]
        self.imputation_method = gui.ControlledList([0], self.listbox)
        gui.button(self.controlArea, self, "Impute", callback=self.comm)


    @Inputs.data
    def set_data(self, dataset):
        if dataset is not None:
            self.dataset = dataset
            self.selection()
        else:
            self.dataset = None
        self.commit()

    def comm(self):
        self.selection()
        self.commit()

    def selection(self):
        if self.dataset is None:
            return
        self.impute()

    def commit(self):
        if self.dataset is None:
            return
        self.Outputs.sample.send(self.dataset)


    def impute(self):
        for dom in self.dataset.domain.variables:
            self.estimate = [d for d in range(len(self.dataset)) if self.dataset[d][dom] == 0]
            totrain =  [d for d in range(len(self.dataset)) if (d not in self.estimate) ]
            self.train = Orange.data.Table.from_table_rows(self.dataset, totrain)
            if(len(self.train)==0):
                continue
            self.gene = dom
            func = self.functions[self.imputation_method[0]]
            eval(func)


    def average(self):
        avg = numpy.mean([d[self.gene] for d in self.train])

        for d in self.estimate:
            self.dataset[d][self.gene] = avg


    def main(argv=None):
        from AnyQt.QtWidgets import QApplication
        # PyQt changes argv list in-place
        app = QApplication(list(argv) if argv else [])
        argv = app.arguments()

        ow = OWDataSamplerA()
        ow.show()
        ow.raise_()

        dataset = Orange.data.Table("ccp_data_Tcells_normCounts.counts.cycle_genes.tab")
        ow.set_data(dataset)
        ow.handleNewSignals()
        app.exec_()
        ow.set_data(None)
        ow.handleNewSignals()
        ow.onDeleteWidget()
        return 0

    if __name__ == "__main__":
        sys.exit(main(sys.argv))

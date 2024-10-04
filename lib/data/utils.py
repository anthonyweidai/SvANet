import os
import numpy as np
import pandas as pd

from ..utils import listPrinter


def initMeanStdByCsv(opt, IsTraining, MeanStdType="train", MeanStdDP=6):
    """ Normalization helps get data within a range and reduces 
    the skewness which helps learn faster and better.
    """
    MeanValues = None
    StdValues = None
    
    if opt.use_meanstd:
        Flag = 0
        DataPath = str(opt.dataset_path) # str(Path(TrainSet[0][0]).parents[1])
        if opt.num_split > 1:
            if os.path.isfile(DataPath + "/norm_mean.csv"):
                Flag = 1
                if "common" not in opt.sup_method and opt.extend_val:
                    # self-sup dataset with split
                    MeanValues = np.zeros(3)
                    StdValues = np.zeros(3)
                    for i in range(opt.num_split):
                        CsvValues = pd.read_csv(DataPath + "/norm_s%d.csv" % (i + 1), index_col=0)
                        MeanValues += np.array(CsvValues[0:1].values.tolist()[0])
                        StdValues += np.array(CsvValues[1:2].values.tolist()[0])
                    # approximation, each split has the close number of images
                    MeanValues = list(MeanValues / opt.num_split)
                    StdValues = list(StdValues / opt.num_split)
                else:
                    CsvValues = pd.read_csv(DataPath + "/norm_mean.csv", index_col=0)
                    MeanValues = CsvValues[opt.split:opt.split + 1].values.tolist()[0]
                    CsvValues = pd.read_csv(DataPath + "/norm_std.csv", index_col=0)
                    StdValues = CsvValues[opt.split:opt.split + 1].values.tolist()[0]
        else:
            if os.path.isfile("%s/norm_%s.csv" % (DataPath, MeanStdType)):
                Flag = 1
                CsvValues = pd.read_csv("%s/norm_%s.csv" % (DataPath, MeanStdType), index_col=0)
                MeanValues = CsvValues[0:1].values.tolist()[0]
                StdValues = CsvValues[1:2].values.tolist()[0]
        
        if MeanValues is not None and StdValues is not None:
            MeanValues = [round(v, MeanStdDP) for v in MeanValues]
            StdValues = [round(v, MeanStdDP) for v in StdValues]
            
        # Norm values printing
        if IsTraining and Flag:
            MeanPrint = ", ".join(map(str, MeanValues))
            StdPrint = ", ".join(map(str, StdValues))
            ListName = ["%s_Mean" % MeanStdType.capitalize(), "%s_Std" % MeanStdType.capitalize()]
            ListValue = [MeanPrint, StdPrint]
            listPrinter(ListName, ListValue)
    
    return MeanValues, StdValues
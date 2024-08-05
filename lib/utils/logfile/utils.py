import os
import re
import csv
from typing import Any

import numpy as np
import pandas as pd
from copy import deepcopy

from ..system import colourText


def listPrinter(ListName, ListValues, LineNum=None, **kwargs: Any):
    Len1 = len(ListName)
    Len2 = len(ListValues)
    assert Len1 == Len2, "Got not consistent len %d and %d" % (Len1, Len2)
    
    if LineNum == None:
        LineNum = 5
    
    Remainder = Len1 % LineNum
    if Remainder > 0 and Remainder <3:
        LineNum -= 1
        Remainder = Len1 % LineNum
        if Remainder > 0 and Remainder < 2:
            LineNum += 2
            Remainder = Len1 % LineNum
            if Remainder > 0 and Remainder < 3:
                LineNum -= 2
                  
    PrintStr = ''
    for Idx in range(Len1):
        if Idx > 0 and Idx % LineNum == 0:
            PrintStr += '\n'
        
        Value = ListValues[Idx]
        if isinstance(Value, float):
            if Value > 1:
                Value = round(Value, 2)
            else:
                Value = round(Value, 4)
        
        PrintStr += '%s: %s, ' % (ListName[Idx], colourText(str(Value), **kwargs))
        # %r for bool printing
    print(PrintStr)


def dictPrinter(DictVar: dict, LineNum=5):
    listPrinter(list(DictVar.keys()), list(DictVar.values()), LineNum)    
    
    
def writeCsv(DestPath, FieldName, FileData, NewFieldNames=[], DictMode=False):
    Flag = 0 if os.path.isfile(DestPath) else 1
    
    with open(DestPath, 'a', encoding='UTF8', newline='') as f:
        if DictMode:
            writer = csv.DictWriter(f, fieldnames=FieldName)
            if Flag == 1:
                writer.writeheader()
            writer.writerows(FileData) # write data
        else:
            writer = csv.writer(f)
            if Flag == 1:
                if NewFieldNames != []:
                    _ = [FieldName.append(FiledName) for FiledName in NewFieldNames]
                writer.writerow(FieldName) # write the header
            writer.writerow(FileData) # write data


def renewCommonLog(RootPath, Exp=None, DropExp: int=None):
    from ..mathematics import averageBestMetrics
    
    LogPath = None
    for file in os.listdir(RootPath):
        if file.endswith(".csv") and "log_" in file:
            LogPath = RootPath + "/" + file # /log_rotation_shuffle.csv # "log_common.csv"
            print("Metadata is %s" % LogPath)
            break
    if LogPath is None:
        return
    
    MetaCsv = pd.read_csv(LogPath)
    Exps = MetaCsv["exp"]
    ExpIds = [list(Exps).index(Exp)] if Exp is not None else range(Exps.__len__())
    
    for e in ExpIds:
        CSVPath = "%s/exp%d/best_metrics.csv" % (RootPath, Exps[e])
        MetricCsv = pd.read_csv(CSVPath)
        ColumnNames = list(MetricCsv.columns.values)
        AvgBestMetric = averageBestMetrics(MetricCsv, AvgSupMode=False)
        
        # for tiny object validation, remove tiny metrics
        TinyIds = [i for i, c in enumerate(ColumnNames) if "tiny" in c.lower()]
        for t in sorted(TinyIds, reverse=True):
            ColumnNames.pop(t)
            AvgBestMetric.pop(t)
        
        for i in range(len(ColumnNames) - 1):
            ColumnName = ColumnNames[i + 1].replace("BestEpoch", "BEpoch").replace("Best", "")
            if ColumnName == "Dice":
                ColumnName = ColumnName.replace("Dice", "mDice")
            MetaCsv.loc[e, ColumnName] = AvgBestMetric[i + 1]
    
    # drop giver exp
    if DropExp is not None:
        DropId = []
        try:
            DropId = list(Exps).index(DropExp)
        except:
            pass
        
        if DropId:
            MetaCsv = MetaCsv.drop(index=DropId)
    
    MetaCsv.to_csv(LogPath, index=False)
    return


def optimiseLogVal(ListValues, Idx=0):
    # remove None, 0, False
    if not ListValues[Idx]:
        ListValues = [ListValues[Idx]] + ['-'] * (len(ListValues) - 1) # ['-' for _ in range(len(ListValues))]
    return ListValues


def createExcelDict(FieldNames: list, ValueList: list) -> dict:
    # use field names and values to create the output excel
    OutputExcel = dict.fromkeys(FieldNames)
    for f, v in zip(FieldNames, ValueList):
        OutputExcel[f] = v
    return OutputExcel


def correctKFoldOrder(KFoldList):
    KFoldNum = [re.findall(r'\d+', k)[0] for k in KFoldList 
                if len(re.findall(r'\d+', k)) > 0]
    
    Unique, Counts = np.unique(KFoldNum, return_counts=True)
    MultipleVal = Unique[Counts > 1]
    if len(MultipleVal) == 0:
        return KFoldList, False
    
    NumUnique = len(Unique[Counts == 1]) + len(MultipleVal)
    
    # a single repeated value
    KFoldNum = np.asarray(KFoldNum)
    CorrectedKFold = deepcopy(KFoldList)
    
    for m in MultipleVal:
        RepeatIds = np.where(KFoldNum == m)[0]
        for i, r in enumerate(RepeatIds):
            if i == 0:
                # keep the first repeated one as origin
                continue
            NumUnique += 1
            CorrectedKFold[r] = 'valrpt_%d' % (NumUnique)
            
    return CorrectedKFold, True


def averageUniqueLog(MetadataGroup, DestPath, DestFileName):
    # average unique column in log file
    UniqueColum = []
    for Name in list(MetadataGroup[0].columns.values):
        if Name in ['Weight']:
            # skip pretrained weight
            continue
        for d in MetadataGroup[1:]:
            if not MetadataGroup[0][Name].equals(d[Name]):
                UniqueColum.append(Name)
                break

    NewMetaCsv = MetadataGroup[0].copy()
    for u in UniqueColum:
        NewMetaCsv[u] = pd.concat([d[u] for d in MetadataGroup], axis=1).mean(axis=1)
    
    DestMeta = '%s/%s.csv' % (DestPath, DestFileName)
    NewMetaCsv.to_csv(DestMeta, index=False)
    print('Destination metadata is %s' % DestMeta)
import os
import csv
from typing import Any

import pandas as pd

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
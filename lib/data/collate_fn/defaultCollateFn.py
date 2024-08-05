import torch
import numpy as np
from typing import List, Dict

from . import registerCollateFn


@registerCollateFn("default")
def defaultCollateFn(Batch: List[Dict]):
    """Default collate function"""
    Keys = list(Batch[0].keys())

    NewBatch = {k: [] for k in Keys}
    for b in Batch:
        for k in Keys:
            NewBatch[k].append(b[k])

    # stack the Keys
    for k in Keys:
        BatchElements = NewBatch.pop(k)
        
        if isinstance(BatchElements[0], (int, float, np.int64)):
            # list of ints or floats
            BatchElements = torch.as_tensor(BatchElements)
        elif isinstance(BatchElements[0], (np.ndarray)):
            BatchElements = torch.from_numpy(BatchElements)
        else:
            # stack tensors (including 0-dimensional)
            try:
                BatchElements = torch.stack(BatchElements, dim=0).contiguous()
            except Exception as e:
                print("Unable to stack the tensors. Error: {}".format(e))
                
        NewBatch[k] = BatchElements

    return NewBatch
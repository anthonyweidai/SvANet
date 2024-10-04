from math import ceil
from typing import Any
import multiprocessing

from .device import CUDA_AVAI
from .utils import colourText
from ..mathematics.utils import makeDivisible


def workerManager(opt, IsPrint=True, **kwargs: Any):
    NUM_PROC = multiprocessing.cpu_count()
    NumWorkers = opt.num_workers
    if NumWorkers is None:
        # if CUDA_AVAI:
        WorkerKanban = opt.batch_size // 4
        NumWorkers = ceil(min(WorkerKanban, NUM_PROC - 2) / 2.) * 2
        NumWorkers = min(makeDivisible(NumWorkers, 4), NUM_PROC)
        # NumWorkers = min(ceil(min(4 * round(NUM_PROC / 8), NUM_PROC - 2) / 2.) * 2, BatchSize)
        
    PinMemory = True if CUDA_AVAI and opt.pin_memory else False
    if IsPrint:
        print("We use [%s/%d] workers, and pin memory is %s" \
            % (colourText(str(NumWorkers)), NUM_PROC, colourText(str(PinMemory))))
    return PinMemory, NumWorkers
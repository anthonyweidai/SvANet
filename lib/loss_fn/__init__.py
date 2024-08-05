from typing import Any

from .segmentation import buildSegmentationLossFn


SupportedTasks = ["segmentation"]

def getLossFn(opt, Task=None, **kwargs: Any):
    if not Task:
        Task = opt.task
    
    LossFn = None
    if Task == "segmentation":
        LossFn = buildSegmentationLossFn(opt, Task=Task, **kwargs) # Task value for mixupMask
    else:
        TaskStr = 'Got {} as a task. Unfortunately, we do not support it yet.' \
                '\nSupported tasks are:'.format(opt.task)
        for i, Name in enumerate(SupportedTasks):
            TaskStr += "\n\t {}: {}".format(i, Name)
    return LossFn
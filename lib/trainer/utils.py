import copy

from torch import nn


def loadWeightFromModel(opt, WeightModel):
    # For storage friendly, damage the performance
    from ..model import getModel
    
    WeightModel = copy.deepcopy(WeightModel.to('cpu'))
    
    Model = getModel(opt=opt).to(opt.device)
    Model.load_state_dict(WeightModel.state_dict())

    return Model


def reparameteriseModel(Model: nn.Module) -> nn.Module:
    """ Method returns a model where a multi-branched structure
        used in training is re-parameterised into a single branch
        for inference.

    :param model: MobileOne model in train mode.
    :return: MobileOne model in inference mode.
    """
    # Avoid editing original graph
    for m in Model.modules():
        if hasattr(m, 'reparameterise'):
            m.reparameterise()
    return Model


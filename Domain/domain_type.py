"Use for other domain type using CNN"
from enum import Enum
class DomainType(str, Enum):
    SOIL = "soil"
    NLP = "nlp"
    SOFWTAREDEFECT = "softwaredefect"

    def __str__(self):
        return self.value

class TransferType(str, Enum):
    Global = "Global"
    Transfer = "Transfer"
    def __str__(self):
        return self.value


class Activation(str, Enum):
    Tanh = "tanh"  #Hyperbolic Tangent
    Relu = "relu" #Rectified Linear Unit
    ELU = "elu"   #Exponential Linear Unit
    SELU = "selu" #Scaled Exponential Linear Unit
    def __str__(self):
        return self.value

class Padding(str, Enum):
    Same = "same"  #same
    Valid = "valid" # no padding mode)

    def __str__(self):
        return self.value



class Optimiser(str, Enum):
    Adams = "Adams"
    SGD ="SGD"
    Adadelta = "Adadelta"
    Adagrad = "Adagrad"
    Adamax = "Adamax"
    Nadam = "Nadam"
    Ftrl = "Ftrl"
    def __str__(self):
        return self.value

'''
kernel_regularizer: Regularizer to apply a penalty on the layer's kernel
bias_regularizer: Regularizer to apply a penalty on the layer's bias
activity_regularizer: Regularizer to apply a penalty on the layer's output
'''
class Regularizer(str, Enum):
    Kernel = "kernel"
    Bias = "bias"
    Activity = "activity"

    def __str__(self):
        return self.value
from yacs.config import CfgNode as CN

_C = CN()
#########################################################################################
# System Parameters
#########################################################################################
_C.SYSTEM = CN()

# Number of GPUS to use in the experiment
_C.SYSTEM.NUM_GPUS = 8

# Number of workers for doing things
_C.SYSTEM.NUM_WORKERS = 4


#########################################################################################
# GENERAL Parameters
#########################################################################################
_C.GENERAL = CN()
_C.GENERAL.SEED =40
_C.GENERAL.LOGS=""
_C.GENERAL.DIRECTORY = "test" # directory path to store information
_C.GENERAL.DOMAIN = "soil"  #domain type  softwaredefect soil or nlp
_C.GENERAL.READCSV = True
_C.GENERAL.READRDS = False
_C.GENERAL.ROOT_DIR = ""
_C.GENERAL.MODEL_DIRECTORY = "" #this will be overwritten inside the train
_C.GENERAL.GLOBAL_MODEL_DIRECTORY =  "" # folder containing the global model to use for this Transfer learning
_C.GENERAL.USING_EXISTING_DATA_SPLIT = False
_C.GENERAL.MODEL_TYPE = ""
#########################################################################################
# Parameters for Data
#########################################################################################
_C.DATA = CN()

_C.DATA.SPECTRA_COLUMN_STARTING= "X"
_C.DATA.PREPROCESSING =  "all"  # the various preprocessing  type are  Absorbances Absorbances-SG1 Absorbances-SG1-SNV Absorbances-SNV-DT' Absorbances-SG0_SNV or all
_C.DATA.ID = ['ID']   # ID in your dataset
_C.DATA.PREDICT_PROPERTIES =['CLAY', 'SILT', 'SAND']  #properties to predict
_C.DATA.OUTPUT_UNIT= 3  # select number of output unit
_C.DATA.MAXNORMALIZATION= 1
_C.DATA.INPUT =  "all"  # the various preprocessing  type are  Absorbances Absorbances-SG1 Absorbances-SG1-SNV Absorbances-SNV-DT' Absorbances-SG0_SNV or all
_C.DATA.STANDAIZATION= "all" #option non, per-band or normalised. standardised. Note both standardised and normalised standards all rows
_C.DATA.GLOBAL_RDATA= ""
_C.DATA.GLOBAL_CSV= [ "" ]
_C.DATA.TARGET_CSV = [ "C:/Projects/ResearchProjects/final_data.csv" ]
_C.DATA.DATA_PARTITION= "DATA/calibration.csv"
_C.DATA.DATA_PARTITION_DEFAULT = "DATA/default-split.csv"
_C.DATA.TARGET_RDATA=""



#########################################################################################
# Parameters for Train
#########################################################################################
_C.TRAIN = CN()
_C.TRAIN.TYPE= "Transfer"   #model type global or transfer
_C.TRAIN.N_REPEATS= 100  #number of times to run this model for prediction to make sure we are not just doing ramdom predction
_C.TRAIN.N_FOLD= 5
_C.TRAIN.VALIDATION_SPLIT = 0.3
_C.TRAIN.CLHS_TRAIN_SIZE = 0.7
_C.TRAIN.HYPERPARAMETER_TYPE = "Population Based"


#########################################################################################
# Parameters for Baysian Hyperparamters
#########################################################################################
_C.BAYESIAN_TUNING= CN()
_C.BAYESIAN_TUNING.INIT_POINT= 10  #number of initial point
_C.BAYESIAN_TUNING.N_ITER= 10  #number of inter
_C.BAYESIAN_TUNING.VERBOSE=1
_C.BAYESIAN_TUNING.DOMAIN_SIZE=1000
_C.BAYESIAN_TUNING.BATCH_SIZE=100
_C.BAYESIAN_TUNING.N_CORES=2
_C.BAYESIAN_TUNING.MAX_NUM_TRIALS=10
_C.BAYESIAN_TUNING.MAX_NUM_CONCURRENT_TRIAL=4


#########################################################################################
# Parameters for POPULATION_BASED_TRAINING
#########################################################################################
_C.POPULATION_BASED_TRAINING= CN()
_C.POPULATION_BASED_TRAINING.POPULATION_SIZE= 2 #number of initial point
_C.POPULATION_BASED_TRAINING.NUM_GENERATIONS= 2  #number of inter
_C.POPULATION_BASED_TRAINING.VERBOSE=2


#########################################################################################
# Parameters for Model Training
#########################################################################################
_C.MODEL = CN()
_C.MODEL.TRANSFER_LEARNING= True
_C.MODEL.FIXED_LAYERS=  4 #number of dense filters
_C.MODEL.PADDING= ["same","valid"]
_C.MODEL.MULTI_TASK= (True, False) # Specify if multi-task learning task-specific output layers
_C.MODEL.ACTIVATION= (1, 4)
_C.MODEL.POOL_SIZE= (1, 2)
_C.MODEL.BATCH_NORM=(True, False)
_C.MODEL.USE_MAXPOOL= (True, False) # USE_MAXPOOL:
_C.MODEL.LAYERS_N= (1, 50)
_C.MODEL.DROP_OUT_LAYER= (True, False)
_C.MODEL.MOMENTUM= (0.0, 0.1)
_C.MODEL.EPSILON= (0.0, 0.0008)
_C.MODEL.REG_SIZE= (0, 0.006)
_C.MODEL.KERNEL_SIZE= (1, 3)
_C.MODEL.REGULARIZER= (1, 3)
_C.MODEL.OPTIMIZER= ("Adams", 'RMSprop', 'SGD', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl')
_C.MODEL.DROP_OUT=(0.0, 0.499)
_C.MODEL.LEARNING_RATE= (0.0, 0.1)
_C.MODEL.NEURON_PCT= (0.01, 0.1)
_C.MODEL.NEURON_SHRINK= (0.01, 0.5)
_C.MODEL.BATCHSIZE= (2, 500)
_C.MODEL.EPOCHS=(1, 200)
_C.MODEL.VERBOSE=1



def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  #"local variable" use pattern
  return _C.clone()


def get_global():
    cfg = _C
    return cfg

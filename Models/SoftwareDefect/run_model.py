import csv
import json
import faulthandler;

from sklearn.model_selection import train_test_split

from Preprocessing.SoftwareDefect.preprocessing import *
faulthandler.enable()
from filelock import FileLock
from sklearn.metrics import balanced_accuracy_score
from ray.util.client import ray
import xgboost as xgb
from ray import tune
from ray.tune.integration.xgboost import TuneReportCheckpointCallback
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.suggest.bohb import TuneBOHB
from Models.SoftwareDefect.Bruakfilter import  *
from Models.SoftwareDefect.HISNN import *
from Models.SoftwareDefect.JDA import *
from Models.SoftwareDefect.TCA import *
from Models.SoftwareDefect.WBDA import *

cfg = data = None


def train_xgboost(config: dict):
    """Function to train a software defect model
       Parameters:
       config: (dict): dictionaries containing the paramters
       Returns:
       Return None
    """

    # set domain adapatation model hyperparamters
    with FileLock(os.path.expanduser("~/.data.lock")):

        # json parameters
        cfg = config['cfg']
        domain_adapation = config['adaptation']
        x_source = config['train_x']
        y_source = config['train_y']
        x_target = config['target_x']
        y_target = config['target_y']

        # set domain adpataion hyperparameters
        print(domain_adapation)
        if domain_adapation == "TCA" or domain_adapation == "JDA":
            kernel_type = config['kernel_type']
            dim = config['dim']
            lamb = config['lamb']
            gamma = config['gamma']
        elif domain_adapation == "Bruakfilter":
            n_neighbors = config['n_neighbors']
        elif domain_adapation == "WBDA" or domain_adapation == "WBDA_PLUS" or domain_adapation == "BDA":
            lamb = config['lamb']
            kernel_type = config['kernel_type']
            dim = config['dim']
            gamma = config['gamma']
            mu = config['mu']

        if domain_adapation == "TCA":
            tca = TCA(dim=dim, lamb=lamb, gamma=gamma, kernel_type=kernel_type)
            x_train, target_x = tca.fit(x_source, x_target)

            y_train, target_y = y_source, y_target
        elif domain_adapation == "JDA":
            jda = JDA(dim=dim, lamb=lamb, gamma=gamma, kernel_type=kernel_type)
            x_train, target_x = jda.fit(x_source, y_source, x_target, y_target)
            y_train, target_y = y_source, y_target
        elif domain_adapation == "Bruakfilter":
            x_train, y_train, target_x, target_y = Bruakfilter(n_neighbors=n_neighbors).run(x_source, y_source,
                                                                                            x_target, y_target)
        elif domain_adapation == "WBDA":
            wbda = BDA(dim=dim, lamb=lamb, mu=mu, mode='WBDA', gamma=gamma, kernel_type=kernel_type)
            print(x_source.dtype.names)
            x_train, target_x = wbda.fit(x_source, y_source, x_target, y_target)
            print(x_train.dtype.names)
            y_train, target_y = y_source, y_target

        elif domain_adapation == "WBDA_PLUS":
            wbda = BDA(dim=dim, lamb=lamb, mu=mu, mode='WBDA_PLUS', gamma=gamma, kernel_type=kernel_type)
            x_train, target_x = wbda.fit(x_source, y_source, x_target, y_target)

            y_train, target_y = y_source, y_target


        # split target data in to 70/30% (70% used to build the transfer model and 30% for testing the model)

        finetune_x, test_x, finetune_y, test_y = train_test_split(target_x, target_y, random_state=cfg.GENERAL.SEED,
                                                                  test_size=cfg.TRAIN.VALIDATION_SPLIT)

        print(finetune_x.shape)

        # concat the source and target fine tune set then split with random seeed
        source_target_train_x = np.concatenate([x_train, finetune_x])
        source_target_train_y = np.concatenate([y_train, finetune_y])

        train_x, val_x, train_y, val_y = train_test_split(source_target_train_x, source_target_train_y,
                                                          random_state=cfg.GENERAL.SEED,
                                                          test_size=cfg.TRAIN.VALIDATION_SPLIT)

        train_set = xgb.DMatrix(train_x, label=train_y)
        val_set = xgb.DMatrix(val_x, label=val_y)

        # now update the target test to be test_x, we will use this for testing
        config['target_test_x'] = test_x  # to old the target test data
        config['target_test_y'] = test_y,

        # Train the base classifier, using the Tune callback
        xgb.train(
            config,
            train_set,
            evals=[(val_set, "eval")],
            verbose_eval=False,
            callbacks=[TuneReportCheckpointCallback(filename="model.xgb")])


def get_best_model_checkpoint(analysis, domain_adapation, x_source, y_source):
    best_bst = xgb.Booster()

    best_bst.load_model(os.path.join(analysis.best_checkpoint, "model.xgb"))

    f_names = best_bst.feature_names
    print(f_names)

    val_accuracy = 1. - analysis.best_result["eval-error"]
    # print(f"Best model parameters: {analysis.best_config}")
    print(f"Best model total accuracy: {val_accuracy:.4f}")

    target_x = analysis.best_config['target_test_x']
    target_y = analysis.best_config['target_test_y']

    # load the best config file

    x_target = analysis.best_config['target_test_x']
    y_target = analysis.best_config['target_test_y']
    kernel_type = analysis.best_config['kernel_type']
    dim = analysis.best_config['dim']
    lamb = analysis.best_config['lamb']
    gamma = analysis.best_config['gamma']
    mu = analysis.best_config['mu']
    n_neighbors = analysis.best_config['n_neighbors']
    KNNneighbors=analysis.best_config['KNNneighbors']
    MinHam=analysis.best_config['MinHam']

    # take only a small subset of the source to transform the test data (for performance only)

    x_source = x_source[:100, :]
    y_source = y_source[:100]

    if domain_adapation == "TCA":
        tca = TCA(dim=dim, lamb=lamb, gamma=gamma, kernel_type=kernel_type)
        x_train, target_x = tca.fit(x_source, x_target)

        y_train, target_y = y_source, y_target
    elif domain_adapation == "JDA":
        jda = JDA(dim=dim, lamb=lamb, gamma=gamma, kernel_type=kernel_type)
        x_train, target_x = jda.fit(x_source, y_source, x_target, y_target)
        y_train, target_y = y_source, y_target
    elif domain_adapation == "Bruakfilter":
        x_train, y_train, target_x, target_y = Bruakfilter(n_neighbors=n_neighbors).run(x_source, y_source,
                                                                                        x_target, y_target)
    elif domain_adapation == "HISNN":
        x_train, y_train, target_x, target_y = HISNN(n_neighbors=n_neighbors,MinHam=MinHam,KNNneighbors=KNNneighbors)\
            .TrainInstanceFiltering(x_source, y_source,x_target, y_target)

    elif domain_adapation == "WBDA":
        wbda = BDA(dim=dim, lamb=lamb, mu=mu, mode='WBDA', gamma=gamma, kernel_type=kernel_type)
        x_train, target_x = wbda.fit(x_source, y_source, x_target, y_target)
        y_train, target_y = y_source, y_target

    elif domain_adapation == "WBDA_PLUS":
        wbda = BDA(dim=dim, mu=mu, mode='WBDA_PLUS', lamb=lamb, gamma=gamma, kernel_type=kernel_type)
        x_train, target_x = wbda.fit(x_source, y_source, x_target, y_target)

        y_train, target_y = y_source, y_target

    test_set = xgb.DMatrix(target_x, label=target_y)

    print(f"Best model total accuracy: {val_accuracy:.4f}")

    pred = best_bst.predict(test_set)


    import numpy as np
    from sklearn.metrics import f1_score

    balanced_acc = balanced_accuracy_score(target_y, np.round(pred))
    f1_score = f1_score(target_y, np.round(pred), average='weighted')
    print(f"Balance acc : {balanced_acc}")
    print(f"f1 score  : {f1_score}")
    '''
    model_xgb_2 = xgb.Booster()
    model_xgb_2.load_model("model.json")
    '''
    return best_bst, val_accuracy, balanced_acc, f1_score


def tune_algorithm(x_source, y_source, target_x, target_y, cfg, domain_adapation, fold):

    dim = lamb = gamma = n_neighbors = mu = kernel_type = MinHam= KNNneighbors=None



    mu = tune.uniform(0.0, 1.0) # # In WBDA, the recommended  setting for mu is 1, lets try grid search
    # set configuration for domain adaptation
    kernel_type = tune.choice(['primal', 'linear', 'rbf'])
    if domain_adapation == "TCA" or domain_adapation == "JDA":
        dim = tune.randint(5, 30)
        lamb = tune.uniform(0.0001, 1)
        gamma = tune.uniform(0.0001, 1)
    elif domain_adapation == "Bruakfilter" or domain_adapation == "HISNN" :
        n_neighbors = tune.randint(1, 100)
        if domain_adapation == "HISNN" :
            MinHam=tune.randint(3, 10)
            KNNneighbors=tune.randint(5, 10)
    elif domain_adapation == "WBDA" or domain_adapation == "WBDA_PLUS" or domain_adapation == "BDA":
        lamb = tune.uniform(0.5, 1)
        dim = tune.randint(5, 30)
        gamma = tune.uniform(0.1, 1)

    # set data
    preProcessing = ['random-over', 'smote', 'random-under', 'none', 'adasyn']

    metricsDir = os.path.join("Result", cfg.GENERAL.DOMAIN, cfg.GENERAL.DIRECTORY, "Metrics")
    modelDir = os.path.join("Result", cfg.GENERAL.DOMAIN, cfg.GENERAL.DIRECTORY, "Model")
    if not os.path.exists(metricsDir):
        os.makedirs(metricsDir)

    if not os.path.exists(modelDir):
        os.makedirs(modelDir)

    file = os.path.join(metricsDir, "metrics-fold.csv")

    for processing_step in preProcessing:
        train_x, train_y = get_data(x_source, y_source, processing_step)

        search_space = {
            # You can mix constants with search space objects.
            "objective": "binary:logistic",
            "eval_metric": ["logloss", "error"],
            "N_estimator": tune.randint(10, 100),
            "learning_rate": tune.loguniform(0.0001, 0.3),
            "max_depth": tune.randint(1, 9),
            "min_child_weight": tune.choice([1, 2, 3]),
            "subsample": tune.uniform(0.5, 1.0),
            "eta": tune.loguniform(1e-4, 1e-1),
            'train_x': train_x,
            'train_y': train_y,
            'target_x': target_x,
            'target_y': target_y,
            'target_test_x': target_x,  # to old the target test data
            'target_test_y': target_y,
            'kernel_type': kernel_type,
            'dim': dim,
            'lamb': lamb,
            'gamma': gamma,
            'MinHam':MinHam,
            'KNNneighbors':KNNneighbors,
            'n_neighbors': n_neighbors,
            'mu': mu,
            'adaptation': domain_adapation,
            'cfg': cfg
        }
        bohb_hyperband = HyperBandForBOHB(
            time_attr="training_iteration",
            max_t=cfg.BAYESIAN_TUNING.MAX_NUM_TRIALS,
            reduction_factor=2,
            # grace_period=4,
            stop_last_trials=False)

        bohb_search = TuneBOHB(
            seed=cfg.GENERAL.SEED,
            # space=config_space,  # If you want to set the space manually
            max_concurrent=cfg.BAYESIAN_TUNING.MAX_NUM_CONCURRENT_TRIAL)

        analysis = tune.run(
            train_xgboost,
            #  server_port=6565,
            name="software defect",
            local_dir=cfg.GENERAL.LOGS,
            metric="eval-logloss",
            mode="min",
            max_failures=7,
            stop={
                "training_iteration": cfg.BAYESIAN_TUNING.MAX_NUM_TRIALS,
                "eval-logloss": 0.4
            },
            # You can add "gpu": 0.1 to allocate GPUs
            resources_per_trial={
                "cpu": 4,
                # By default, Tune automatically runs N concurrent trials, where N is the number of CPUs (cores) on your machine.
                "gpu": 0.1
            },
            num_samples=cfg.BAYESIAN_TUNING.N_ITER,  # Number of times to sample from the hyperparameter space
            config=search_space,
            search_alg=bohb_search,
            scheduler=bohb_hyperband)

        # plot trials
        # ax = None  # This plots everything on the same plot
        # for d in analysis.values():
        #     ax = d.logloss.plot(ax=ax, legend=False)
        # ax.show()

        # path to save model config
        model_config_path = os.path.join(modelDir, "{0}--best-config--{1}.json".format(domain_adapation,
                                                                                       processing_step))

        # path to save model
        model_path = os.path.join(modelDir,
                                  "{0}--model-{1}.json".format(domain_adapation, processing_step))

        # # save best config config to report on
        best_config = str(analysis.best_config)
        with open(model_config_path, 'w') as fp:
            json.dump(best_config, fp)

        best_bst, val_accuracy, balanced_acc, f1_score = get_best_model_checkpoint(analysis, domain_adapation, train_x,
                                                                                   train_y)

        # tune the base model and repeats

        # save model
        best_bst.save_model(model_path)
        # save the result to file
        with open(file, 'a', newline='') as csvfile:
            fieldnames = ['classifier_name', 'dataset_name', 'val_accuracy', 'BUC', 'F1', 'Fold']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writerow({'classifier_name': domain_adapation, 'dataset_name': processing_step,
                             'val_accuracy': val_accuracy, 'BUC': balanced_acc, 'F1': f1_score, 'Fold': fold})


def get_data(x, y, preProcessing):
    if preProcessing == 'smote':
        x, y = smote(x, y)
    elif preProcessing == 'randomover':
        x, y = randomOverSampling(x, y)
    elif preProcessing == 'randomunder':
        x, y = randomUnderSampling(x, y)
    elif preProcessing == 'adasyn':
        x, y = adasyn(x, y)
    elif preProcessing == 'none':
        x, y = x, y
    return x, y


def run_model(m_cfg, m_data):
    global cfg
    global data
    cfg = m_cfg
    data = m_data

    # for nesi
    ray.init(num_cpus=64, num_gpus=1)

    # ray.init(num_cpus=8)

    x_train = data.Xsource
    y_train = data.Ysource
    x_target = data.Xtarget
    y_target = data.Ytarget

    adaptation = ['JDA', 'TCA', 'Bruakfilter', 'WBDA', 'WBDA_PLUS','HISNN']
    for t in range(cfg.TRAIN.N_REPEATS):
        for domain_adapation in adaptation:
            tune_algorithm(x_train, y_train, x_target, y_target, cfg, domain_adapation, t)





import pytest

import numpy as np
from sklearn import linear_model

import kerastuner


INPUT_DIM = 2
NUM_CLASSES = 3
NUM_SAMPLES = 64
TRAIN_INPUTS = np.random.random(size=(NUM_SAMPLES, INPUT_DIM))
TRAIN_TARGETS = np.random.randint(0, NUM_CLASSES, size=(NUM_SAMPLES,))
VAL_INPUTS = np.random.random(size=(NUM_SAMPLES, INPUT_DIM))
VAL_TARGETS = np.random.randint(0, NUM_CLASSES, size=(NUM_SAMPLES,))


class SklearnModel(object):

    def __init__(self, model):
        self.model = model

    def fit(self, x, y, validation_data):
        self.model.fit(x, y)
        x_val, y_val = validation_data
        return self.model.score(x_val, y_val)


def sklearn_build_fn(hp):
    penalty = hp.Choice('penalty', ['l1', 'l2'])
    c = hp.Float('c', 1e-4, 10)
    model = linear_model.LogisticRegression(penalty=penalty, C=c)
    return SklearnModel(model)


def test_base_tuner_build_fn():
    tuner = kerastuner.engine.base_tuner.BaseTuner(
        kerastuner.tuners.randomsearch.RandomSearchOracle(),
        sklearn_build_fn,
        max_trials=2,
        executions_per_trial=3)

    assert tuner.max_trials == 2
    assert tuner.executions_per_trial == 3
    assert tuner.hypermodel.__class__.__name__ == 'DefaultHyperModel'
    assert len(tuner.hyperparameters.space) == 2  # default search space
    assert len(tuner.hyperparameters.values) == 2  # default search space

    tuner.search(x=TRAIN_INPUTS,
                 y=TRAIN_TARGETS,
                 validation_data=(VAL_INPUTS, VAL_TARGETS))

    assert len(tuner.trials) == 2
    assert tuner.trials[0].score > 0
    assert tuner.trials[1].score > 0

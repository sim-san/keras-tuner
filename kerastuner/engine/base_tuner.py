# Copyright 2019 The Keras Tuner Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"Abstract tuner base interface."

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import time

import numpy as np

from . import hyperparameters as hp_module
from . import hypermodel as hm_module
from . import oracle as oracle_module
from . import trial as trial_module


class BaseTuner(object):

    def __init__(self,
                 oracle,
                 hypermodel,
                 max_trials,
                 executions_per_trial=1,
                 hyperparameters=None,
                 tune_new_entries=True,
                 allow_new_entries=True):
        """Abstract BaseTuner class reusable for non-Keras models.

        May be subclassed to create new tuners.

        Args:
            oracle: Instance of Oracle class.
            hypermodel: Instance of HyperModel class
                (or callable that takes hyperparameters
                and returns a Model instance).
            objective: String. Name of model metric to minimize
                or maximize, e.g. "val_accuracy".
            max_trials: Int. Total number of trials
                (model configurations) to test at most.
                Note that the oracle may interrupt the search
                before `max_trial` models have been tested.
            executions_per_trial: Int. Number of executions
                (training a model from scratch,
                starting from a new initialization)
                to run per trial (model configuration).
                Model metrics may vary greatly depending
                on random initialization, hence it is
                often a good idea to run several executions
                per trial in order to evaluate the performance
                of a given set of hyperparameter values.
            hyperparameters: HyperParameters class instance.
                Can be used to override (or register in advance)
                hyperparamters in the search space.
            tune_new_entries: Whether hyperparameter entries
                that are requested by the hypermodel
                but that were not specified in `hyperparameters`
                should be added to the search space, or not.
                If not, then the default value for these parameters
                will be used.
            allow_new_entries: Whether the hypermodel is allowed
                to request hyperparameter entries not listed in
                `hyperparameters`.
            """
        if not isinstance(oracle, oracle_module.Oracle):
            raise ValueError('Expected oracle to be '
                             'an instance of Oracle, got: %s' % (oracle,))
        self.oracle = oracle
        if isinstance(hypermodel, hm_module.HyperModel):
            self.hypermodel = hypermodel
        else:
            if not callable(hypermodel):
                raise ValueError(
                    'The `hypermodel` argument should be either '
                    'a callable with signature `build(hp)` returning a model, '
                    'or an instance of `HyperModel`.')
            self.hypermodel = hm_module.DefaultHyperModel(hypermodel)

        # Global search options
        self.max_trials = max_trials
        self.executions_per_trial = executions_per_trial

        # Search space management
        if hyperparameters:
            self.hyperparameters = hyperparameters
            self._initial_hyperparameters = hyperparameters.copy()
        else:
            self.hyperparameters = hp_module.HyperParameters()
            self._initial_hyperparameters = None
            if not tune_new_entries:
                raise ValueError(
                    'If you set `tune_new_entries=False`, you must'
                    'specify the search space via the '
                    '`hyperparameters` argument.')
            if not allow_new_entries:
                raise ValueError(
                    'If you set `allow_new_entries=False`, you must'
                    'specify the search space via the '
                    '`hyperparameters` argument.')
        self.tune_new_entries = tune_new_entries
        self.allow_new_entries = allow_new_entries

        # Public internal state.
        # This contains the results of the search.
        self.trials = []

        # Populate initial search space
        if not self.hyperparameters.space and self.tune_new_entries:
            self.hypermodel.build(self.hyperparameters)

    def generate_trial_id(self):
        # Note: this could be obtained from an oracle.
        # The random ID implementation below is a placeholder.
        return str(random.randint(1e6, 1e7))

    def search(self, *fit_args, **fit_kwargs):
        self.on_search_begin()
        for _ in range(self.max_trials):
            # Obtain unique trial ID to communicate with the oracle.
            trial_id = self.generate_trial_id()
            hp = self._call_oracle(trial_id)
            if hp is None:
                # Oracle triggered exit
                return
            # The Trial ID will contain the results (hp values and score).
            trial = trial_module.Trial(
                trial_id=trial_id,
                hyperparameters=hp.copy(),
                num_executions=self.executions_per_trial,
            )
            self.trials.append(trial)
            self.on_trial_begin(trial)
            self.run_trial(trial, hp, fit_args, fit_kwargs)
            self.on_trial_end(trial)
        self.on_search_end()

    def run_trial(self, trial, hp, fit_args, fit_kwargs):
        """Trains a model (possibly multiple times) and compute a trial score.

        After this method returns, the property `trial.score` should be set to
        a scalar (by default, lower is better).

        Args:
            trial: Instance of `Trial`.
            hp: Instance of `HyperParameters` containing a set of values
                to evaluate.
            fit_args: Arguments to pass to `model.fit`.
            fit_kwargs: Keyword arguments to pass to `model.fit`.
        """
        scores = []
        for i in range(self.executions_per_trial):
            if not self.tune_new_entries:
                # Make copy of hp to avoid mutating it
                hp = hp.copy()

            model = self.hypermodel.build(hp)

            score = model.fit(*fit_args, **fit_kwargs)
            scores.append(score)
        trial.score = np.mean(scores)

    def on_search_begin(self):
        pass

    def on_trial_begin(self, trial):
        pass

    def on_trial_end(self, trial):
        pass

    def on_search_end(self):
        pass

    def _call_oracle(self, trial_id):
        if not self.tune_new_entries:
            # In this case, never append to the space
            # so work from a copy of the internal hp object
            hp = self._initial_hyperparameters.copy()
        else:
            # In this case, append to the space,
            # so pass the internal hp object to `build`
            hp = self.hyperparameters

        # Obtain hp value suggestions from the oracle.
        while 1:
            oracle_answer = self.oracle.populate_space(trial_id, hp.space)
            if oracle_answer['status'] == 'RUN':
                hp.values = oracle_answer['values']
                return hp
            elif oracle_answer['status'] == 'EXIT':
                print('Oracle triggered exit')
                return
            else:
                time.sleep(10)

from typing import Dict

import pyspiel
from open_spiel.python import policy
from open_spiel.python.algorithms import exploitability
from pdcfrplus.utils.logger import Logger

from pdcfrplus.game import GameConfig


class StateBase:
    id = 0

    def __init__(self, h: pyspiel.State):
        StateBase.id += 1
        self.id = StateBase.id + 1
        self.legal_actions = h.legal_actions()
        self.player = h.current_player()
        self.feature = h.information_state_string(self.player)
        self.num_actions = len(self.legal_actions)
        self.children = {a: [] for a in self.legal_actions}
        self.max_utility, self.min_utility = -1e5, 1e5
        self.init_data()

    def init_data(self):
        self.policy = self.init_uniform_policy()
        self.cum_policy = {a: 0 for a in self.legal_actions}

    def get_average_policy(self):
        cum_sum = sum(self.cum_policy.values())
        ave_policy = {}
        for a, cum in self.cum_policy.items():
            if cum_sum == 0:
                ave_policy[a] = 1 / self.num_actions
            else:
                ave_policy[a] = cum / cum_sum
        return ave_policy

    def init_uniform_policy(self):
        uniform_policy = {a: 1 / self.num_actions for a in self.legal_actions}
        return uniform_policy

    def __str__(self):
        return "{}/{}".format(self.feature, str(self.player))


class SolverBase:
    def __init__(self, game_config: GameConfig, logger: Logger = None):
        self.game_config = game_config
        if logger is None:
            logger = Logger(writer_strings=[])
        self.logger = logger
        self.game = game_config.load_game()
        self.states: Dict[str, StateBase] = {}
        self.num_iteration = 0
        self.num_players = self.game.num_players()
        self.total_iterations = self.game_config.iterations
        self.exps = [0 for _ in range(self.total_iterations + 1)]
        self.init_states(self.game.new_initial_state())

    def learn(self, eval_interval: int = 1):
        self.evaluate()
        while self.num_iteration < self.total_iterations:
            # for self.num_iteration in range(1, self.total_iterations + 1):
            self.iteration()
            if self.num_iteration % eval_interval == 0:
                self.evaluate()

    def evaluate(self):
        exp = self.calc_exp()
        # state_dict = self.get_state_dict()
        # self.logger.record("state_dict", state_dict)
        self.logger.record("exp", exp)
        self.logger.record("iter", self.num_iteration)
        self.logger.dump(step=self.num_iteration)
        self.exps[self.num_iteration] = exp

    def get_state_dict(self):
        raise NotImplemented

    def iteration(self):
        raise NotImplemented

    def init_states(self, h: pyspiel.State):
        if h.is_terminal():
            return
        if h.is_chance_node():
            for a in h.legal_actions():
                self.init_states(h.child(a))
            return
        self.lookup_state(h, h.current_player())
        for a in h.legal_actions():
            self.init_states(h.child(a))

    def lookup_state(self, h: pyspiel.State, player: int) -> StateBase:
        feature = h.information_state_string(player)
        feature = self.add_player_info_in_feature(feature, player)
        if self.states.get(feature) is None:
            self.states[feature] = self.init_state(h)
        return self.states[feature]

    def add_player_info_in_feature(self, feature, player):
        feature = feature + "/" + str(player)
        return feature

    def init_state(self, h: pyspiel.State):
        return StateBase(h)

    def calc_exp(self):
        exp = exploitability.exploitability(
            self.game,
            policy.tabular_policy_from_callable(self.game, self.average_policy()),
        )
        exp = max(exp, 1e-12)
        return exp

    def average_policy(self):
        def wrap(h):
            feature = h.information_state_string()
            player = h.current_player()
            feature = self.add_player_info_in_feature(feature, player)
            s = self.states[feature]
            return s.get_average_policy()

        return wrap

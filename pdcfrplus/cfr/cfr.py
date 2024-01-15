import numpy as np
from pdcfrplus.utils.logger import Logger

from pdcfrplus.cfr.cfr_base import SolverBase, StateBase
from pdcfrplus.game import GameConfig


class CFRState(StateBase):
    def init_data(self):
        super().init_data()
        self.reach = 0
        self.imm_regrets = {a: 0 for a in self.legal_actions}
        self.imm_regrets_copy = {a: 0 for a in self.legal_actions}
        self.regrets = {a: 0 for a in self.legal_actions}

    def update_regret(self):
        for a in self.legal_actions:
            self.regrets[a] = self.regrets[a] + self.imm_regrets[a]

    def update_current_policy(self):
        regret_sum = 0
        for regret in self.regrets.values():
            regret_sum += max(0, regret)
        for a, regret in self.regrets.items():
            if regret_sum == 0:
                self.policy[a] = 1 / self.num_actions
            else:
                self.policy[a] = max(0, regret) / regret_sum

    def cumulate_policy(self, T, gamma):
        for a, p in self.policy.items():
            if T == 1:
                self.cum_policy[a] = self.reach * p
                continue
            self.cum_policy[a] = (
                self.cum_policy[a] * np.power((T - 1) / T, gamma) + self.reach * p
            )

    def clear_temp(self):
        for a in self.regrets.keys():
            self.imm_regrets_copy[a] = self.imm_regrets[a]
            self.imm_regrets[a] = 0
        self.reach = 0


class CFR(SolverBase):
    def __init__(self, game_config: GameConfig, logger: Logger = None, gamma: int = 0):
        super().__init__(game_config, logger)
        self.gamma = gamma

    def init_state(self, h):
        return CFRState(h)

    def get_state_dict(self):
        state_dict = {}
        state_dict["states"] = {}
        for feature, state in self.states.items():
            state_policy = state.policy
            state_regrets = state.regrets
            state_imm_regrets = state.imm_regrets_copy
            state_cum_policy = state.cum_policy
            state_ave_policy = state.get_average_policy()
            state_dict["states"][feature] = {
                "policy": state_policy,
                "regrets": state_regrets,
                "cum_policy": state_cum_policy,
                "ave_policy": state_ave_policy,
                "imm_regret": state_imm_regrets,
            }
        state_dict["iteration"] = self.num_iteration

        return state_dict

    def iteration(self):
        self.num_iteration += 1
        for i in range(self.num_players):
            self.clear_temp(i)
            h = self.game.new_initial_state()
            self.calc_regret(h, i, 1, 1)

            pending_states = [s for s in self.states.values() if s.player == i]

            for s in pending_states:
                self.update_state(s)

    def calc_regret(self, h, traveser, my_reach, opp_reach):
        if h.is_terminal():
            return h.returns()[traveser]

        if h.is_chance_node():
            v = 0
            for a, p in h.chance_outcomes():
                v += p * self.calc_regret(h.child(a), traveser, my_reach, opp_reach * p)
            return v

        cur_player = h.current_player()
        s = self.lookup_state(h, cur_player)

        if cur_player != traveser:
            v = 0
            for a in h.legal_actions():
                p = s.policy[a]
                v += p * self.calc_regret(h.child(a), traveser, my_reach, opp_reach * p)
            return v

        child_v = {}
        v = 0
        for a in h.legal_actions():
            p = s.policy[a]
            child_v[a] = self.calc_regret(h.child(a), traveser, my_reach * p, opp_reach)
            v += p * child_v[a]

        for a in h.legal_actions():
            s.imm_regrets[a] += opp_reach * (child_v[a] - v)

        s.reach += my_reach
        return v

    def clear_temp(self, player):
        for state in self.states.values():
            if state.player == player:
                state.clear_temp()

    def update_state(self, s):
        s.update_regret()

        s.cumulate_policy(self.num_iteration, self.gamma)
 
        s.update_current_policy()

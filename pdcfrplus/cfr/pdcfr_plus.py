import numpy as np

from pdcfrplus.cfr.cfr import CFR, CFRState


class PDCFRPlusState(CFRState):
    def weight(self, T, alpha):
        d, w = np.power(T - 1, alpha) / (np.power(T - 1, alpha) + 1), 1
        return d, w

    def update_regret(self, T, alpha):
        d, w = self.weight(T, alpha)
        for a in self.legal_actions:
            self.regrets[a] = max(self.regrets[a] * d + self.imm_regrets[a] * w, 0)

    def update_current_policy(self, T, alpha):
        d, w = self.weight(T + 1, alpha)
        self.pred_regrets = {}
        for a in self.legal_actions:
            self.pred_regrets[a] = max(self.regrets[a] * d + self.imm_regrets[a] * w, 0)
        regret_sum = 0
        for regret in self.pred_regrets.values():
            regret_sum += max(0, regret)
        for a, regret in self.pred_regrets.items():
            if regret_sum == 0:
                self.policy[a] = 1 / self.num_actions
            else:
                self.policy[a] = max(0, regret) / regret_sum


class PDCFRPlus(CFR):
    def __init__(self, game_config, logger=None, gamma=5, alpha=2.3):
        super().__init__(game_config, logger, gamma)
        self.alpha = alpha

    def init_state(self, h):
        return PDCFRPlusState(h)

    def update_state(self, s):
        s.update_regret(self.num_iteration, self.alpha)

        s.cumulate_policy(self.num_iteration, self.gamma)

        s.update_current_policy(self.num_iteration, self.alpha)

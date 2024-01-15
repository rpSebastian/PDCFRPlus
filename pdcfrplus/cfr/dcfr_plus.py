import numpy as np

from pdcfrplus.cfr.cfr import CFR, CFRState


class DCFRPlusState(CFRState):
    def update_regret(self, T, alpha=1.5):
        T = float(T)
        for a in self.legal_actions:
            w = np.power(T - 1, alpha) / (np.power(T - 1, alpha) + 1.5)
            self.regrets[a] = max(self.regrets[a] * w + self.imm_regrets[a], 0)


class DCFRPlus(CFR):
    def __init__(self, game_config, logger=None, gamma=4, alpha=1.5):
        super().__init__(game_config, logger, gamma)
        self.alpha = alpha

    def init_state(self, h):
        return DCFRPlusState(h)

    def update_state(self, s):
        s.update_regret(self.num_iteration, self.alpha)

        s.cumulate_policy(self.num_iteration, self.gamma)

        s.update_current_policy()

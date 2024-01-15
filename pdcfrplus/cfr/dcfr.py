import numpy as np

from pdcfrplus.cfr.cfr import CFR, CFRState


class DCFRState(CFRState):
    def update_regret(self, T, alpha=1.5, beta=0):
        T = float(T)
        for a in self.legal_actions:
            if T == 1:
                self.regrets[a] = self.imm_regrets[a]
                continue
            if self.regrets[a] > 0:
                self.regrets[a] = (
                    self.regrets[a]
                    * (np.power(T - 1, alpha) / (np.power(T - 1, alpha) + 1))
                    + self.imm_regrets[a]
                )
            else:
                self.regrets[a] = (
                    self.regrets[a]
                    * (np.power(T - 1, beta) / (np.power(T - 1, beta) + 1))
                    + self.imm_regrets[a]
                )


class DCFR(CFR):
    def __init__(self, game_config, logger=None, alpha=1.5, beta=0, gamma=2):
        super().__init__(game_config, logger, gamma)
        self.alpha = alpha
        self.beta = beta

    def init_state(self, h):
        return DCFRState(h)

    def update_state(self, s):
        s.update_regret(self.num_iteration, self.alpha, self.beta)

        s.cumulate_policy(self.num_iteration, self.gamma)

        s.update_current_policy()

from pdcfrplus.cfr.cfr import CFR, CFRState


class CFRPlusState(CFRState):
    def update_regret(self):
        for a in self.legal_actions:
            self.regrets[a] = max(self.regrets[a] + self.imm_regrets[a], 0)


class CFRPlus(CFR):
    def __init__(self, game_config, logger=None, gamma=1):
        super().__init__(game_config, logger, gamma)

    def init_state(self, h):
        return CFRPlusState(h)

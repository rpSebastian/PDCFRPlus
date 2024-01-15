from PokerRL.game import bet_sets
from PokerRL.game.games import (
    DiscretizedNLHoldemSubGame3,
    DiscretizedNLHoldemSubGame4,
    StandardLeduc,
)
from PokerRL.rl.base_cls.workers.ChiefBase import ChiefBase
from pdcfrplus.utils.utils import init_object, load_module


class CFRRunner:
    def __init__(
        self,
        algo_name,
        game_name,
        iterations,
        logger,
        alpha=1.5,
        gamma=0,
        beta=1,
    ):
        self.solver_class = load_module("PokerRL.cfr:{}".format(algo_name))
        game_dict = {
            "Subgame3": DiscretizedNLHoldemSubGame3,
            "Subgame4": DiscretizedNLHoldemSubGame4,
            "Leduc": StandardLeduc,
        }
        self.game_class = game_dict[game_name]
        self.iterations = iterations
        self.logger = logger
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        chief = ChiefBase(t_prof=None)
        config = dict(
            name="test",
            game_cls=self.game_class,
            agent_bet_set=bet_sets.B_3,
            other_agent_bet_set=bet_sets.B_2,
            chief_handle=chief,
            alpha=self.alpha,
            gamma=self.gamma,
            beta=self.beta,
        )
        self.cfr = init_object(self.solver_class, config)
        self.step = 0

    def run(self):
        for _ in range(1, self.iterations + 1):
            self.iteration()
            self.evaluate()

    def iteration(self):
        self.step += 1
        self.cfr.iteration()

    def evaluate(self):
        self.conv = self.cfr.expl / 1000
        if self.step == 1:
            self.logger.record(f"exp", self.conv)
            self.logger.record(f"iter", 0)
            self.logger.dump(step=0)
        self.logger.record(f"exp", self.conv)
        self.logger.record(f"iter", self.step)
        self.logger.dump(step=self.step)

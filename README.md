# Minimizing Weighted Counterfactual Regret with Optimistic Online Mirror Descent

> Minimizing Weighted Counterfactual Regret with Optimistic Online Mirror Descent <br>
> Hang Xu, Kai Li<sup>\#</sup>, Bingyun Liu, Haobo Fu, Qiang Fu, Junliang Xing<sup>#</sup>, Jian Cheng <br>
> IJCAI 2024 (Oral)

## Install PDCFRPlus

Install miniconda3 from [the official website](https://docs.conda.io/en/latest/miniconda.html) and run the following script:

```bash
bash scripts/install.sh
```

## Test PDCFRPlus

We use games implemented by [OpenSpiel](https://github.com/deepmind/open_spiel) [1] and [PokerRL](https://github.com/EricSteinberger/PokerRL) [2]. Run the following script to assess the performance of CFR variants on testing games. The results are saved in the folder `results`.
```bash
conda activate PDCFRPlus
python scripts/parallel_run.py --algo CFRPlus
python scripts/parallel_run.py --algo LinearCFR
python scripts/parallel_run.py --algo DCFR
python scripts/parallel_run.py --algo PCFRPlus --gamma=2
python scripts/parallel_run.py --algo PCFRPlus --gamma=5
python scripts/parallel_run.py --algo DCFRPlus --gamma=4 --alpha=1.5
python scripts/parallel_run.py --algo PDCFRPlus --gamma=5 --alpha=2.3
```

## References

[1] Lanctot, M.; Lockhart, E.; Lespiau, J.-B.; Zambaldi, V.; Upadhyay, S.; P´erolat, J.; Srinivasan, S.; Timbers, F.; Tuyls, K.; Omidshafiei, S.; Hennes, D.; Morrill, D.; Muller, P.; Ewalds, T.; Faulkner, R.; Kram´ar, J.; Vylder, B. D.; Saeta, B.; Bradbury, J.; Ding, D.; Borgeaud, S.; Lai, M.; Schrittwieser, J.; Anthony, T.; Hughes, E.; Danihelka, I.; and Ryan-Davis, J. 2019. OpenSpiel: A Framework for Reinforcement Learning in Games. CoRR, abs/1908.09453.

[2] Steinberger, E. 2019. PokerRL. https://github.com/TinkeringCode/PokerRL.

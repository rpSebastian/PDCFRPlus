# Minimizing Weighted Counterfactual Regret with Optimistic Online Mirror Descent

> Minimizing Weighted Counterfactual Regret with Optimistic Online Mirror Descent <br>
> Hang Xu<sup>\*</sup> , Kai Li<sup>\*,\#</sup>, Bingyun Liu, Haobo Fu, Qiang Fu, Junliang Xing<sup>#</sup> <br>
> IJCAI 2024

## Install PDCFRPlus

Install miniconda3 from [the official website](https://docs.conda.io/en/latest/miniconda.html) and run the following script:

```bash
bash scripts/install.sh
```

## Test PDCFRPlus

Run the following script to assess the performance of CFR variants on testing games. The results are saved in the folder `results`.
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

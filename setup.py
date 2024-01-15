import os

from setuptools import find_packages, setup

this_dir = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_dir, "requirements.txt"), "r") as f:
    requirements = [line.strip() for line in f.readlines()]

setup(
    name="pdcfrplus",
    version="1.0.0",
    description="PDCFR+: Minimizing Weighted Counterfactual Regret with Optimistic Online Mirror Descent",
    packages=find_packages(exclude=("tests*", "docs*", "examples*", "repos*")),
    zip_safe=True,
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)

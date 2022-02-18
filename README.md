# Assessing the Robustness of Quantum Ridge Regression on DWave's Quantum Annealers

**Authors**: Alessandro Sanvito, Thuany K. Stuart

## Abstract

Machine learning has become the de-facto standard for modeling complex real-world phenomena, with relevant implications across several industries. However, with the increasing complexity of the models, the greater availability of data, and Moore's law decline have put pressure on researchers to find suitable accelerators to support machine learning computations.

Among the candidates, quantum computers, and quantum annealers, in particular, can provide theoretical speed-ups to machine learning computations, but the current hardware suffers from errors in the results due to thermal noise and structural limitations.

In this paper, we report the results of the empirical analysis of a classical machine learning model, Ridge regression, trained on a DWave's hybrid solver in a realistic scenario. First, we provide a formulation of the training algorithm suitable for the quantum annealer. Then we analyze the results of the comparison between quantum training instances and models trained with the Cholesky closed-form solution to assess the robustness of the training process across different datasets.

Overall, our work finds that training Ridge regression on a quantum annealer does not yield the same reliable results as on classical hardware.

[Link to the paper](paper.pdf)

## How to

In order to run the experiments, delete the elements in the /data folter and from the main directory execute ./python3 run.py.

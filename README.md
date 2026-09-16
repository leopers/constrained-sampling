# Constrained Sampling with Primal-Dual Langevin Monte Carlo

A PyTorch implementation of Algorithm 1 from _Constrained Sampling with Primal-Dual Langevin Monte Carlo_, with experiments based on Figure 2 of the paper.

## Overview

This project explores sampling from a Gaussian distribution under support constraints using primal-dual Langevin Monte Carlo (PD-LMC). The implementation combines Langevin sampling with adaptive Lagrange multipliers to control expected constraint violations.

## Main Experiment

The main experiment adapts the two-dimensional Gaussian sampling example from Figure 2. The unit disk is replaced by an axis-aligned ellipse centered at the origin, with semi-axis lengths of \(0.5\) along the horizontal axis and \(2\) along the vertical axis.

The authors’ JAX implementation serves as a reference but this project implements the algorithm in PyTorch.

## Extension

### Square Support

An additional experiment explores a square support region, including the formulation of suitable constraints and an analysis of the resulting samples.

## References

- [paper](https://openreview.net/pdf?id=o6Hk6vld20)
- [two-dimensional Gaussian implementation](https://github.com/lfochamon/pdlmc/blob/main/support_constraint/2d_gaussian.py)
- [experiment notebook](https://github.com/lfochamon/pdlmc/blob/main/support_constraint/2d_gaussian.ipynb)

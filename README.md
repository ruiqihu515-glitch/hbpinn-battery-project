# HB-PINN Battery

A Python package for reproducing and extending a hybrid physics-informed workflow for Li-ion battery prognosis.

This project is inspired by the paper:

**A framework for Li-ion battery prognosis based on hybrid Bayesian physics-informed neural networks**  
Nascimento, Viana, Corbetta, and Kulkarni, Scientific Reports, 2023.

## Project idea

The goal of this project is to build a small but complete Python package for battery prognosis.

The package will include:

1. NASA battery data loading and preprocessing.
2. A physics-informed voltage model.
3. A calibrated hybrid surrogate model.
4. Battery aging modelling with uncertainty.
5. KL-divergence based model-health monitoring.
6. Reproducible figures for presentation and reporting.

## Current status

This project is being built step by step from scratch.

## Folder structure

```text
hbpin_battery_project/
├── src/
│   └── hbpinn_battery/
│       └── __init__.py
├── README.md
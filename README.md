# Project Overview

This repository contains a research-oriented Python toolkit for differential privacy (DP) analysis on medical imaging and patient metadata, specifically using the OASIS brain MRI dataset. The project integrates Laplace noise mechanisms, decision tree modeling, and sensitivity analysis to evaluate the privacy-utility trade-off in machine learning for healthcare.

The toolkit includes utilities for data preprocessing, noise injection, model training/testing, and privacy evaluation. It also supports synthetic data generation and external dataset integration (e.g., Citiesp.xlsx for geolocation data).

# Project Structure

- utilities.py           # Core utility functions (file I/O, noise injection, data trimming)
- sensitivity_epsilon.py # DP analysis, model building, sensitivity/entropy calculations
- research_plan.py       # Main research pipeline for answering predefined questions
- training.py            # OASIS dataset loading, patient/session data handling
- flatten.py             # Binary file flattening and hex representation
- test.py                # Synthetic data generation with Laplace noise (using Citiesp.xlsx)
- Citiesp.xlsx           # External dataset for location-based synthetic data

# Key Features

## Differential Privacy Tools:
- Laplace noise injection with adjustable epsilon
- Bounded/unbounded DP sensitivity analysis
- Entropy calculations per data dimension

## Machine Learning Integration:
- Decision tree classifier (scikit-learn)
- Training/test splitting, accuracy evaluation
- Model serialization/deserialization

## Data Handling:
- OASIS metadata and MRI image processing
- Synthetic patient data generation
- Location-based data from Citiesp.xlsx

## Research Automation:
- Predefined research questions
- Automated accuracy/sensitivity reporting
- Interactive model selection and testing

# Research Questions Addressed
- Accuracy of baseline model on filled data
- Accuracy of baseline model on noisy data
- Sensitivity of accuracy under unbounded DP
- Sensitivity of accuracy under bounded DP
- Entropy measurements per dataset dimension

# Key Functions
- laplace_noise(): Adds Laplace noise for DP
- get_dt_unbounded_privacy(): Computes sensitivity for unbounded DP
- get_dt_bounded_privacy(): Computes sensitivity for bounded DP
- build_decision_tree(): Trains a decision tree on OASIS data
- add_noise_files() / add_fill_files(): Generates DP-augmented metadata files


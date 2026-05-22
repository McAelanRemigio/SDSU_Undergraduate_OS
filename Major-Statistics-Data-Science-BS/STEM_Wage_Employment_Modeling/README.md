# STEM Employment and Wage Modeling Case Study

## Overview

This project analyzes U.S. STEM occupation employment and wage data from 2020–2024 to explore how employment levels relate to annual median wages.

The original academic project was developed as part of an upper-division data science course. This public version has been revised for portfolio use: it focuses on methodology, interpretation, and applied learning rather than reproducing a class submission.

## Research Question

How did employment levels and wages change across selected U.S. STEM occupations from 2020 to 2024, and how well can employment-related features predict annual median wage?

## Dataset

The analysis uses U.S. occupational employment and wage data for STEM-related occupations from 2020–2024.

Key variables include:

- `TOT_EMP`: total employment
- `A_MEDIAN`: annual median wage
- `Year`: reporting year
- occupation group indicators derived from occupational codes

## Methods

The workflow includes:

1. Data cleaning and preparation
2. Exploratory data analysis
3. Baseline model construction
4. Simple linear regression using employment as the main predictor
5. Train/test evaluation
6. Feature engineering, including log transformations and occupation-group interaction terms
7. Model comparison using RMSE
8. Reflection on model limitations and future improvements

## Key Findings

The analysis found that total employment alone was a weak predictor of annual median wage. The relationship between employment and wages was highly variable, with substantial differences across STEM occupation groups.

Feature engineering slightly improved model performance, but the improvement was limited. This suggested that wage prediction requires richer contextual features beyond employment counts, such as industry, education requirements, specialization, experience level, or regional labor market conditions.

## What This Project Demonstrates

This project demonstrates my ability to:

- work with real-world public datasets
- clean and prepare structured data
- use EDA to guide modeling decisions
- evaluate model performance with train/test splits
- apply feature engineering thoughtfully
- interpret weak model results honestly
- explain why more advanced methods may be needed

## Limitations

The model relies on occupation-level summary data, which limits its ability to explain wage differences. STEM wages are influenced by many variables not included in the dataset, including industry, geography, education, credentials, and seniority.

Future extensions could include:

- random forest regression
- richer occupation-level feature engineering
- regional wage comparisons
- industry-specific wage modeling
- comparison between linear and nonlinear models

## Academic Integrity Note

This repository is intended as a portfolio case study and learning artifact. It does not include exams, graded submissions, or instructional materials. The public notebook has been revised from the original academic version to emphasize concepts, methodology, and interpretation.

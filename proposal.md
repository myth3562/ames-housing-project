# Project Proposal: Ames Housing Data Analysis

## 1. Research Question
How accurately can we predict the sale price of a residential property based on its physical characteristics? This project evaluates whether regularized regression models (Ridge/Lasso) outperform a standard baseline Multiple Linear Regression model when handling categorical neighborhoods and square footage metrics.

## 2. The Dataset
* **Dataset Name:** Ames Housing Dataset
* **Source Platform:** [Kaggle Ames Housing Competition](https://kaggle.com)
* **Description:** Contains 79 explanatory variables describing residential homes in Ames, Iowa. 

## 3. Two-Source Scan of Existing Work
* **Source 1: Journal of Statistics Education (De Cock, 2011):** Highlights that data preprocessing—particularly removing specific outliers (homes > 4,000 sq ft with low sale prices)—is critical for stabilizing linear regression models.
* **Source 2: Kaggle Notebook Community Consensus:** Demonstrates that converting nominal variables via one-hot encoding, paired with Lasso regression for automated feature selection, yields optimal Root Mean Squared Error (RMSE).

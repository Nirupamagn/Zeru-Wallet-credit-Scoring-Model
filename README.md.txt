Zeru Wallet credit Scoring Model

 ## Summary

Based on past transaction activity, the DeFi Wallet Credit Scoring Model seeks to assign a credit score to wallets via the Aave V2 protocol.  Traditional credit scoring techniques are inapplicable in the decentralized finance (DeFi) environment, thus creating a model that assesses wallet responsibility and dependability is crucial.

## Goal

Assigning each wallet a credit score between 0 and 1000 is the main objective of this project; higher ratings denote responsible usage, while lower scores represent dangerous or exploitative behavior.  Users and protocols can evaluate the creditworthiness of wallets in the DeFi domain with the use of this rating system.


## Features
Transaction-level data from the Aave V2 protocol, comprising activities like deposits, borrows, repays, and liquidations, is used as the data source.
The concept of "feature engineering"  identifies important characteristics in transaction data, such as: - Total USD deposits and borrowingsThe regularity and recentness of transactions; risk markers such liquidation occurrences
- Diverse assets in deals
The methodology used for scoring:  assigns credit scores based on cluster features and uses K-Means, DBSCAN, and Gaussian Mixture Models (GMM) to classify wallets according to their behavior.

Install the required libraries:
pip install -r requirements.txt

Run the scoring script:
python score_wallets.py


The results will be saved in CSV files named 
scored_wallets_kmeans.csv, 
scored_wallets_dbscan.csv, and 
scored_wallets_gmm.csv.
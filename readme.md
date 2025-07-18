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

## Features Engineered
- Total Transactions: Count of all transactions per wallet.
- Unique Assets: Number of unique assets interacted with.
- Transaction Amounts: Total amount deposited, borrowed, repaid, etc.
- Transaction Frequency: Average time between transactions.
- Last Transaction Age: Days since the last transaction.
- Action Types: Breakdown of actions (deposit, borrow, repay, etc.).
- Behavior Ratios: Ratios such as repay-to-borrow ratio.

## Machine Learning Algorithms Used
- K-Means: For clustering wallets based on transaction behavior.
- DBSCAN: For identifying clusters and noise points.
- Gaussian Mixture Model (GMM): For probabilistic clustering.

## Processing Flow
1. Load Data: Read the JSON file containing transaction data.
2. Feature Engineering: Create features based on the transaction data.
3. Clustering: Use multiple machine learning algorithms to cluster wallets based on engineered features.
4. Score Calculation: Assign credit scores based on cluster assignments and behavior metrics.
5. Output Results: Save the scored wallets to CSV files.
6. Visualization: Generate a score distribution graph.

Install the required libraries:
pip install -r requirement.txt

Run the scoring script:
python score_wallets.py


The results will be saved in CSV files named 
scored_wallets_kmeans.csv, 
scored_wallets_dbscan.csv, and 
scored_wallets_gmm.csv.

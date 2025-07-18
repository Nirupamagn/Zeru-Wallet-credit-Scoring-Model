import json
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

def load_data(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        print("DataFrame columns:", df.columns)  # Print columns for debugging
        return df
    except FileNotFoundError:
        print(f"Error: The file {filepath} was not found.")
        return pd.DataFrame()  # Return an empty DataFrame
    except json.JSONDecodeError:
        print(f"Error: The file {filepath} is not a valid JSON.")
        return pd.DataFrame()  # Return an empty DataFrame

def engineer_features(df):
    # Convert timestamp to datetime
    df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='s')

    # Extract fields from actionData
    df['actionData.amount'] = pd.to_numeric(df['actionData'].apply(lambda x: x.get('amount', 0) if isinstance(x, dict) else 0), errors='coerce')
    df['actionData.assetPriceUSD'] = pd.to_numeric(df['actionData'].apply(lambda x: x.get('assetPriceUSD', 0) if isinstance(x, dict) else 0), errors='coerce')
    df['actionData.assetSymbol'] = df['actionData'].apply(lambda x: x.get('assetSymbol', '') if isinstance(x, dict) else '')

    # Calculate USD value for each transaction
    df['amount_usd'] = df['actionData.amount'] * df['actionData.assetPriceUSD']

    # Group by userWallet to engineer features
    wallet_features = df.groupby('userWallet').agg(
        total_transactions=('txHash', 'count'),
        first_tx_time=('timestamp_dt', 'min'),
        last_tx_time=('timestamp_dt', 'max'),
        num_unique_assets=('actionData.assetSymbol', 'nunique')
    ).reset_index()

    wallet_features['wallet_age_days'] = (wallet_features['last_tx_time'] - wallet_features['first_tx_time']).dt.days

    # Aggregate features for different actions
    for action_type in ['deposit', 'borrow', 'repay', 'redeemunderlying', 'liquidationcall']:
        action_df = df[df['action'] == action_type]
        if not action_df.empty:
            action_agg = action_df.groupby('userWallet').agg(
                total_amount_usd=('amount_usd', 'sum'),
                num_actions=(f'action', 'count'),
                avg_amount_usd=('amount_usd', 'mean'),
                first_action_time=('timestamp_dt', 'min'),
                last_action_time=('timestamp_dt', 'max')
            ).reset_index()
            action_agg.columns = [f'{action_type}_{col}' if col not in ['userWallet'] else col for col in action_agg.columns]
            wallet_features = pd.merge(wallet_features, action_agg, on='userWallet', how='left')

            # Calculate frequency and recency for each action
            if f'{action_type}_num_actions' in wallet_features.columns and wallet_features[f'{action_type}_num_actions'].sum() > 0:
                wallet_features[f'{action_type}_frequency_days'] = (wallet_features[f'{action_type}_last_action_time'] - wallet_features[f'{action_type}_first_action_time']).dt.days / wallet_features[f'{action_type}_num_actions']
                wallet_features[f'{action_type}_last_action_age_days'] = (datetime.now() - wallet_features[f'{action_type}_last_action_time']).dt.days

    # Fill NaN values for wallets that didn't perform certain actions
    wallet_features = wallet_features.fillna(0)

    # Calculate derived ratios
    wallet_features['repay_to_borrow_ratio_usd'] = wallet_features['repay_total_amount_usd'] / wallet_features['borrow_total_amount_usd']
    wallet_features['repay_to_borrow_ratio_usd'] = wallet_features['repay_to_borrow_ratio_usd'].replace([np.inf, -np.inf], 0).fillna(0)  # Handle inf and NaN

    wallet_features['liquidation_ratio'] = wallet_features['liquidationcall_num_actions'] / wallet_features['borrow_num_actions']
    wallet_features['liquidation_ratio'] = wallet_features['liquidation_ratio'].replace([np.inf, -np.inf], 0).fillna(0)  # Handle inf and NaN

    # Select features for clustering
    features_for_clustering = [
        'total_transactions', 'wallet_age_days', 'num_unique_assets',
        'deposit_total_amount_usd', 'deposit_num_actions', 'deposit_frequency_days', 'deposit_last_action_age_days',
        'borrow_total_amount_usd', 'borrow_num_actions', 'borrow_frequency_days', 'borrow_last_action_age_days',
        'repay_total_amount_usd', 'repay_num_actions', 'repay_frequency_days', 'repay_last_action_age_days',
        'repay_to_borrow_ratio_usd',
        'liquidationcall_num_actions', 'liquidation_ratio'
    ]
    
    # Filter out features that might not exist if no transactions of that type occurred in the dataset
    features_for_clustering = [f for f in features_for_clustering if f in wallet_features.columns]

    return wallet_features, features_for_clustering

def assign_credit_scores(wallet_features_df, features_for_clustering, algorithm='kmeans', n_clusters=5):
    # Scale features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(wallet_features_df[features_for_clustering])
    
    if algorithm == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    elif algorithm == 'dbscan':
        model = DBSCAN(eps=0.5, min_samples=5)
    elif algorithm == 'gmm':
        model = GaussianMixture(n_components=n_clusters, random_state=42)
    else:
        raise ValueError("Unsupported algorithm. Choose 'kmeans', 'dbscan', or 'gmm'.")

    # Fit the model
    wallet_features_df['cluster'] = model.fit_predict(scaled_features)

    # Calculate behavior_proxy after clustering
    wallet_features_df['behavior_proxy'] = (
        wallet_features_df['repay_to_borrow_ratio_usd'] * 1000  # Amplify good behavior
        - wallet_features_df['liquidation_ratio'] * 5000  # Penalize liquidations heavily
        + wallet_features_df['deposit_total_amount_usd'] / 1e9  # Scale large amounts
    )

    # Determine the number of unique clusters
    unique_clusters = wallet_features_df['cluster'].unique()
    num_clusters = len(unique_clusters)

    # Adjust for DBSCAN noise points
    if algorithm == 'dbscan':
        unique_clusters = unique_clusters[unique_clusters != -1]
        num_clusters = len(unique_clusters)

    # Check if there are any valid clusters
    if num_clusters == 0:
        print("No valid clusters found. Assigning default credit scores.")
        wallet_features_df['credit_score'] = 0  # Assign a default score
        return wallet_features_df[['userWallet', 'credit_score', 'cluster']]

    # Create score ranges based on the number of clusters found
    score_ranges = np.linspace(0, 1000, num_clusters + 1)
    cluster_to_score_map = {cluster: (score_ranges[i], score_ranges[i + 1]) for i, cluster in enumerate(unique_clusters)}

    # Assign a score within the cluster's range based on individual wallet's behavior_proxy
    wallet_features_df['credit_score'] = 0.0
    for cluster_id, (min_score, max_score) in cluster_to_score_map.items():
        cluster_wallets = wallet_features_df[wallet_features_df['cluster'] == cluster_id]
        
        if not cluster_wallets.empty:
            min_proxy = cluster_wallets['behavior_proxy'].min()
            max_proxy = cluster_wallets['behavior_proxy'].max()
            
            if max_proxy == min_proxy:  # Handle case where all proxies in cluster are the same
                wallet_features_df.loc[wallet_features_df['cluster'] == cluster_id, 'credit_score'] = (min_score + max_score) / 2
            else:
                # Linear interpolation within the cluster's score range
                wallet_features_df.loc[wallet_features_df['cluster'] == cluster_id, 'credit_score'] = \
                    min_score + (max_score - min_score) * (cluster_wallets['behavior_proxy'] - min_proxy) / (max_proxy - min_proxy)
    
    # Ensure scores are within 0-1000 and are integers
    wallet_features_df['credit_score'] = wallet_features_df['credit_score'].clip(0, 1000).astype(int)

    return wallet_features_df[['userWallet', 'credit_score', 'cluster']]

def visualize_score_distribution(wallet_features_df):
    plt.figure(figsize=(10, 6))
    
    # Define score ranges
    bins = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    labels = ['0-100', '100-200', '200-300', '300-400', '400-500', 
              '500-600', '600-700', '700-800', '800-900', '900-1000']
    
    # Create a copy of the DataFrame to avoid SettingWithCopyWarning
    df_copy = wallet_features_df.copy()
    
    # Create a histogram
    df_copy['score_range'] = pd.cut(df_copy['credit_score'], bins=bins, labels=labels, right=False)
    score_distribution = df_copy['score_range'].value_counts().sort_index()
    
    # Plotting
    score_distribution.plot(kind='bar', color='blue', alpha=0.7)
    plt.title('Distribution of Wallet Credit Scores')
    plt.xlabel('Score Ranges')
    plt.ylabel('Number of Wallets')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.75)
    plt.savefig('score_distribution.png')  # Save the figure
    plt.show()

if __name__ == "__main__":
    filepath = 'user-wallet-transactions.json'  # Adjust path as needed
    
    print("Loading data...")
    df = load_data(filepath)
    
    print("Engineering features...")
    wallet_features_df, features_for_clustering = engineer_features(df)
    
    print("Assigning credit scores using K-Means...")
    scored_wallets_kmeans = assign_credit_scores(wallet_features_df, features_for_clustering, algorithm='kmeans', n_clusters=10)
    
    print("Assigning credit scores using DBSCAN...")
    scored_wallets_dbscan = assign_credit_scores(wallet_features_df, features_for_clustering, algorithm='dbscan')
    
    print("Assigning credit scores using GMM...")
    scored_wallets_gmm = assign_credit_scores(wallet_features_df, features_for_clustering, algorithm='gmm', n_clusters=10)
    
    print("\nSample Scored Wallets (K-Means):")
    print(scored_wallets_kmeans.head())
    
    print("\nSample Scored Wallets (DBSCAN):")
    print(scored_wallets_dbscan.head())
    
    print("\nSample Scored Wallets (GMM):")
    print(scored_wallets_gmm.head())
    
    # Visualize score distribution
    visualize_score_distribution(scored_wallets_kmeans)

    # Save results
    scored_wallets_kmeans.to_csv('scored_wallets_kmeans.csv', index=False)
    scored_wallets_dbscan.to_csv('scored_wallets_dbscan.csv', index=False)
    scored_wallets_gmm.to_csv('scored_wallets_gmm.csv', index=False)
    print("\nScored wallets saved to scored_wallets_kmeans.csv, scored_wallets_dbscan.csv, and scored_wallets_gmm.csv")

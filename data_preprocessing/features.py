import pandas as pd

class Features:

    def __init__(self):

        self.features = []
        self.feature_idx = []

    def extract_features(self,feature_path: str):

        """
        Extract all possible features from dataframe into arrays.
        """

        features_df = pd.read_csv(feature_path)
        self.features = features_df["feature_text"].tolist()
        self.feature_idx = features_df["feature_num"].tolist()

    def get(self, idx):

        """
        Get feature and its corresponding idx of dataframe.

        """
        return self.features[idx],self.feature_idx[idx]

    def len(self):

        """
        Return length
        """

        return len(self.features)



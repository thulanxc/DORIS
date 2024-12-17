from pathlib import Path
import pickle
import logging
from typing import Dict, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
import yaml
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)

class EmbeddingGenerator:
    """Generate embeddings for all posts."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.model = SentenceTransformer(self.config['embedding_model']['name'])
        
    def process_dataset(self, input_path: str, output_path: str) -> None:
        """Generate embeddings for a dataset."""
        # Load data
        df = pd.read_csv(input_path)
        
        # Generate embeddings
        embeddings = []
        for text in tqdm(df['text'], desc="Generating embeddings"):
            embedding = self.model.encode(text)
            embeddings.append(embedding)
            
        # Add embeddings to DataFrame
        df['embedding'] = embeddings
        
        # Save with embeddings
        with open(output_path, 'wb') as f:
            pickle.dump(df, f)

class PostHistoryExtractor:
    """Extract post history representations from existing embeddings."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.output_dir = Path(self.config['paths']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compute_post_history_representation(self, user_posts: pd.DataFrame) -> np.ndarray:
        """Compute average embedding across all user's posts."""
        post_embeddings = np.stack(user_posts['embedding'].values)
        return np.mean(post_embeddings, axis=0)

    def process_dataset(self, input_path: str, output_path: str) -> None:
        """Process dataset with existing embeddings."""
        logging.info(f"Processing dataset: {input_path}")
        
        # Load data with embeddings
        with open(input_path, 'rb') as f:
            df = pickle.load(f)
        
        # Process each user
        post_history_features = {}
        for user_id, user_posts in tqdm(df.groupby('user_id'),
                                      desc="Computing post history representations"):
            post_history_features[user_id] = self.compute_post_history_representation(user_posts)
        
        # Save results
        with open(output_path, 'wb') as f:
            pickle.dump(post_history_features, f)
        
        logging.info(f"Saved post history features to {output_path}")
        logging.info(f"Processed {len(post_history_features)} users")

def main():
    """Main execution function."""
    # First, generate embeddings for all datasets
    embedding_generator = EmbeddingGenerator('config.yaml')
    
    for dataset in ['train', 'test']:
        # Generate embeddings
        embedding_generator.process_dataset(
            input_path=f"data/{dataset}_data.csv",
            output_path=f"data/{dataset}_with_emb.pkl"
        )
        
        # Extract post history features using generated embeddings
        extractor = PostHistoryExtractor('config.yaml')
        extractor.process_dataset(
            input_path=f"data/{dataset}_with_emb.pkl",
            output_path=f"results/{dataset}_post_history_features.pkl"
        )

if __name__ == "__main__":
    main()
import logging
from pathlib import Path
import pickle
from typing import List, Dict, Union, Tuple, Optional
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import torch
from tqdm import tqdm
from multiprocessing import Pool
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)

class DepressionFeatureExtractor:
    """Extract depression-related features from social media posts using embedding similarity and LLM."""
    
    def __init__(
        self,
        config_path: str = 'config.yaml',
        device: Optional[str] = None
    ):
        """
        Initialize the feature extractor.
        
        Args:
            config_path: Path to configuration file
            device: Computing device ('cuda' or 'cpu')
        """
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.embedding_model = SentenceTransformer(
            self.config['embedding_model']['name']
        ).to(self.device)
        
        # Initialize OpenAI client if LLM processing is enabled
        if self.config['llm']['enabled']:
            self.client = OpenAI(api_key=self.config['llm']['api_key'])
        
        # Create output directories
        self.output_dir = Path(self.config['paths']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_symptom_templates(self) -> List[str]:
        """Load symptom description templates from file."""
        with open(self.config['paths']['templates'], 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines()]

    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Compute embeddings for a list of texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            np.ndarray: Matrix of embeddings
        """
        return self.embedding_model.encode(
            texts,
            batch_size=self.config['embedding_model']['batch_size'],
            show_progress_bar=True,
            convert_to_numpy=True
        )

    def calculate_similarities(
        self,
        post_embeddings: np.ndarray,
        template_embeddings: np.ndarray
    ) -> pd.DataFrame:
        """
        Calculate cosine similarities between posts and templates.
        
        Args:
            post_embeddings: Matrix of post embeddings
            template_embeddings: Matrix of template embeddings
            
        Returns:
            pd.DataFrame: Similarity scores for each template
        """
        similarities = cosine_similarity(post_embeddings, template_embeddings)
        
        # Create DataFrame with similarity scores
        similarity_df = pd.DataFrame(
            similarities,
            columns=[f'sim_{i+1}' for i in range(template_embeddings.shape[0])]
        )
        similarity_df['average_sim'] = similarity_df.mean(axis=1)
        
        return similarity_df

    def process_post_llm(self, args: Tuple[int, Dict]) -> Tuple[int, str, str]:
        """Process a single post with LLM."""
        idx, data = args
        prompt = self._create_diagnostic_prompt(data['text'])
        response = self._get_llm_completion(prompt)
        return idx, data['hash'], response

def _create_diagnostic_prompt(self, text: str) -> str:
   """Create prompt for diagnostic criteria assessment based on DSM-5 criteria."""
   return f"""Assuming you are a psychiatrist specializing in depression. Given the following text, please determine if this message includes any of the following states of the author:

A. Depressed mood
B. Loss of interest/pleasure  
C. Weight loss or gain
D. Insomnia or hypersomnia
E. Psychomotor agitation or retardation
F. Fatigue
G. Inappropriate guilt
H. Decreased concentration
I. Thoughts of suicide

Text: {text}

If present, answer in the format of enclosed letters separated by commas, for example, (A, B, C). If none are present, respond with None."""

    def _get_llm_completion(self, prompt: str) -> str:
        """Get completion from LLM with retry logic."""
        max_retries = self.config['llm']['max_retries']
        for attempt in range(max_retries):
            try:
                completion = self.client.chat.completions.create(
                    model=self.config['llm']['model_name'],
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0
                )
                return completion.choices[0].message.content
            except Exception as e:
                if attempt == max_retries - 1:
                    logging.error(f"Failed after {max_retries} attempts: {str(e)}")
                    return "Error"
                time.sleep(2 ** attempt)

    def process_dataset(
        self,
        input_path: str,
        output_path: str,
        is_training: bool = True
    ) -> None:
        """
        Process a complete dataset through the pipeline.
        
        Args:
            input_path: Path to input dataset
            output_path: Path to save processed features
            is_training: Whether this is training data
        """
        # Load data
        with open(input_path, 'rb') as f:
            df = pickle.load(f)
            
        # Load templates
        templates = self.load_symptom_templates()
        
        # Compute embeddings
        template_embeddings = self.compute_embeddings(templates)
        post_embeddings = np.stack(df['embedding'].values)
        
        # Calculate similarities
        similarity_df = self.calculate_similarities(post_embeddings, template_embeddings)
        df = pd.concat([df, similarity_df], axis=1)
        
        # Process high-risk posts with LLM if enabled
        if self.config['llm']['enabled']:
            threshold = self.config['llm']['similarity_threshold']
            high_risk_df = df[df['average_sim'] > threshold].copy()
            
            with Pool(processes=self.config['llm']['batch_size']) as pool:
                results = []
                for result in tqdm(
                    pool.imap_unordered(self.process_post_llm, high_risk_df.iterrows()),
                    total=len(high_risk_df)
                ):
                    results.append(result)
            
            # Convert LLM results to binary features
            llm_df = self._convert_to_binary_features(pd.DataFrame(
                results, columns=['index', 'hash', 'expert']))
                
            # Merge back with original data
            df = pd.merge(df, llm_df, on='hash', how='left')
        
        # Save processed features
        df.to_pickle(output_path)

    def _convert_to_binary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert LLM outputs to binary features."""
        for col in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
            df[col] = 0
            
        for idx, row in df.iterrows():
            if "None" not in str(row['expert']):
                letters = str(row['expert']).replace('(', '').replace(')', '').replace(' ', '').split(',')
                for letter in letters:
                    if letter in df.columns:
                        df.at[idx, letter] = 1
        
        return df[['hash', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']]

def main():
    """Main execution function."""
    # Initialize extractor
    extractor = DepressionFeatureExtractor('config.yaml')
    
    # Process train and test sets
    for dataset in ['train', 'test']:
        extractor.process_dataset(
            input_path=f"data/{dataset}_with_emb.pkl",
            output_path=f"results/{dataset}_symptom_features.pkl",
            is_training=(dataset == 'train')
        )

if __name__ == "__main__":
    main()
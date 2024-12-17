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
import yaml
from datetime import datetime

class MoodCourseExtractor:
    """Extract mood course representations from user posts."""
    
    def __init__(
        self,
        config_path: str = 'config.yaml',
        device: Optional[str] = None
    ):
        """
        Initialize the mood course extractor.
        
        Args:
            config_path: Path to configuration file
            device: Computing device ('cuda' or 'cpu')
        """
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_model = SentenceTransformer(
            self.config['embedding_model']['name']
        ).to(self.device)
        self.client = OpenAI(api_key=self.config['llm']['api_key'])
        
        # Create output directories
        self.output_dir = Path(self.config['paths']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_emotion_templates(self) -> Dict[str, str]:
        """Load emotion templates from file."""
        with open(self.config['paths']['emotion_templates'], 'r', encoding='utf-8') as f:
            templates = yaml.safe_load(f)
        return templates

    def compute_embeddings(self, texts: List[str]) -> np.ndarray:
        """Compute embeddings for texts."""
        return self.embedding_model.encode(
            texts,
            batch_size=self.config['embedding_model']['batch_size'],
            show_progress_bar=True,
            convert_to_numpy=True
        )

    def filter_emotional_posts(
        self,
        posts: pd.DataFrame,
        template_embeddings: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """Filter posts with high emotional content."""
        all_emotional_posts = set()
        
        # Get post embeddings
        post_embeddings = np.stack(posts['embedding'].values)
        
        # Calculate similarities for each emotion
        for emotion, template_emb in template_embeddings.items():
            similarities = cosine_similarity(post_embeddings, [template_emb])[..., 0]
            
            # Select top m% posts for each emotion
            threshold = np.percentile(
                similarities,
                100 - self.config['mood_course']['emotion_threshold_percentile']
            )
            emotional_indices = np.where(similarities >= threshold)[0]
            all_emotional_posts.update(emotional_indices)

        return posts.iloc[list(all_emotional_posts)].copy()

    def create_mood_course_prompt(self, posts: pd.DataFrame) -> str:
        """Create prompt for mood course analysis."""
        # Sort posts by timestamp
        sorted_posts = posts.sort_values('timestamp')
        
        # Format posts with timestamps
        post_sequence = []
        for _, row in sorted_posts.iterrows():
            timestamp = datetime.fromtimestamp(row['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            post_sequence.append(f"Time: {timestamp}, Post: {row['text']}")
        
        post_text = "\n".join(post_sequence)
        
        return f"""As a consulting psychiatrist, please conduct a longitudinal mood course analysis based on the following temporal sequence of personal expressions. For each entry, evaluate affect, emotional valence, and severity of mood states. Synthesize these observations into a clinical summary of mood progression, noting any patterns of persistence, fluctuation, or changes over time:

{post_text}"""

    def get_mood_course_summary(self, prompt: str) -> str:
        """Get mood course summary from LLM."""
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

    def compute_mood_course_representation(
        self,
        summary_embedding: np.ndarray,
        post_embeddings: np.ndarray
    ) -> np.ndarray:
        """Compute final mood course representation."""
        alpha = self.config['mood_course']['alpha']
        beta = self.config['mood_course']['beta']
        
        average_post_embedding = np.mean(post_embeddings, axis=0)
        
        return alpha * summary_embedding + beta * average_post_embedding

    def process_user(self, user_data: pd.DataFrame) -> np.ndarray:
        """Process single user's posts to get mood course representation."""
        # Load emotion templates and compute embeddings
        emotion_templates = self.load_emotion_templates()
        template_embeddings = {
            emotion: self.compute_embeddings([template])[0]
            for emotion, template in emotion_templates.items()
        }
        
        # Filter emotional posts
        emotional_posts = self.filter_emotional_posts(user_data, template_embeddings)
        
        if len(emotional_posts) == 0:
            return np.zeros(self.config['embedding_model']['embedding_dim'])
        
        # Get mood course summary
        prompt = self.create_mood_course_prompt(emotional_posts)
        summary = self.get_mood_course_summary(prompt)
        
        # Compute final representation
        summary_embedding = self.compute_embeddings([summary])[0]
        post_embeddings = np.stack(emotional_posts['embedding'].values)
        
        return self.compute_mood_course_representation(summary_embedding, post_embeddings)

    def process_dataset(
        self,
        input_path: str,
        output_path: str,
    ) -> None:
        """Process complete dataset to extract mood course features."""
        # Load data
        with open(input_path, 'rb') as f:
            df = pickle.load(f)
        
        # Process each user
        mood_representations = {}
        for user_id, user_posts in tqdm(df.groupby('user_id')):
            mood_representations[user_id] = self.process_user(user_posts)
        
        # Save results
        with open(output_path, 'wb') as f:
            pickle.dump(mood_representations, f)

def main():
    """Main execution function."""
    extractor = MoodCourseExtractor('config.yaml')
    
    for dataset in ['train', 'test']:
        extractor.process_dataset(
            input_path=f"data/{dataset}_with_emb.pkl",
            output_path=f"results/{dataset}_mood_course_features.pkl"
        )

if __name__ == "__main__":
    main()
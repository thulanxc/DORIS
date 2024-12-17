from pathlib import Path
import logging
import pickle
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from openai import OpenAI
import yaml
from tqdm import tqdm
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)

class ExplanationGenerator:
    """Generate explanations for depression detection results."""
    
    def __init__(self, config_path: str = 'config.yaml'):
        """
        Initialize the explanation generator.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.config['llm']['api_key'])
        
        # Create output directories
        self.output_dir = Path(self.config['paths']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def format_symptom_posts(self, posts_with_symptoms: pd.DataFrame) -> str:
        """
        Format posts and their symptoms for the explanation prompt.
        
        Args:
            posts_with_symptoms: DataFrame containing posts and their symptoms
            
        Returns:
            str: Formatted post and symptom information
        """
        formatted_posts = []
        for _, row in posts_with_symptoms.iterrows():
            symptoms = [col for col in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I'] 
                       if row[col] == 1]
            if symptoms:
                post_info = f"Post: '{row['text']}'\nSymptoms: ({', '.join(symptoms)})"
                formatted_posts.append(post_info)
        
        return "\n\n".join(formatted_posts)

    def create_explanation_prompt(
        self,
        mood_course: str,
        symptom_posts: str,
        is_depressed: bool
    ) -> str:
        """
        Create prompt for generating explanation.
        
        Args:
            mood_course: Mood course description
            symptom_posts: Formatted symptom posts
            is_depressed: Classification result
            
        Returns:
            str: Formatted prompt
        """
        classification = "depressed" if is_depressed else "normal"
        
        return f"""Assuming you are a psychiatrist specializing in depression.

Here is a user's mood course: {mood_course}

Below are posts from this user displaying symptoms of depression and the types of symptoms exhibited:
{symptom_posts}

This user has been determined by an automated depression detection system to be {classification}.

Please consider the user's mood course and posts to generate an explanation for this judgment. Your explanation should be grounded in concrete evidence."""

    def get_explanation(self, prompt: str) -> str:
        """
        Get explanation from LLM with retry logic.
        
        Args:
            prompt: Input prompt
            
        Returns:
            str: Generated explanation
        """
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
                    return "Error generating explanation"
                time.sleep(2 ** attempt)

    def process_user(
        self,
        user_id: str,
        posts: pd.DataFrame,
        mood_course: str,
        prediction: bool
    ) -> Dict:
        """
        Generate explanation for a single user.
        
        Args:
            user_id: User identifier
            posts: User's posts with symptom annotations
            mood_course: User's mood course description
            prediction: Model's prediction (True for depressed)
            
        Returns:
            Dict: User ID, prediction, and explanation
        """
        symptom_posts = self.format_symptom_posts(posts)
        prompt = self.create_explanation_prompt(mood_course, symptom_posts, prediction)
        explanation = self.get_explanation(prompt)
        
        return {
            'user_id': user_id,
            'prediction': prediction,
            'explanation': explanation
        }

    def process_dataset(
        self,
        predictions_path: str,
        mood_course_path: str,
        posts_path: str,
        output_path: str
    ) -> None:
        """
        Process complete dataset to generate explanations.
        
        Args:
            predictions_path: Path to model predictions
            mood_course_path: Path to mood course descriptions
            posts_path: Path to posts with symptom annotations
            output_path: Path to save results
        """
        logging.info("Starting explanation generation process")
        
        # Load data
        predictions = pd.read_csv(predictions_path)
        with open(mood_course_path, 'rb') as f:
            mood_courses = pickle.load(f)
        with open(posts_path, 'rb') as f:
            posts_df = pickle.load(f)
        
        # Generate explanations for each user
        explanations = []
        for _, row in tqdm(predictions.iterrows(), desc="Generating explanations"):
            user_id = row['user_id']
            user_posts = posts_df[posts_df['user_id'] == user_id]
            
            explanation = self.process_user(
                user_id=user_id,
                posts=user_posts,
                mood_course=mood_courses[user_id],
                prediction=row['prediction']
            )
            explanations.append(explanation)
        
        # Save results
        results_df = pd.DataFrame(explanations)
        results_df.to_csv(output_path, index=False)
        
        logging.info(f"Saved explanations to {output_path}")
        logging.info(f"Processed {len(explanations)} users")

def main():
    """Main execution function."""
    explainer = ExplanationGenerator('config.yaml')
    
    for dataset in ['train', 'test']:
        explainer.process_dataset(
            predictions_path=f"results/{dataset}_predictions.csv",
            mood_course_path=f"results/{dataset}_mood_course.pkl",
            posts_path=f"data/{dataset}_with_symptoms.pkl",
            output_path=f"results/{dataset}_explanations.csv"
        )

if __name__ == "__main__":
    main()
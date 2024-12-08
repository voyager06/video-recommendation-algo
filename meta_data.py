#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel

# Initialize the FastAPI app
app = FastAPI()

# Helper function to generate tags based on category
def _generate_tags(category_id):
    tag_collections = {
        1: ['innovation', 'tech', 'startup', 'ai', 'coding'],  # Technology
        2: ['fitness', 'football', 'training', 'athletes', 'competition'],  # Sports
        3: ['movies', 'music', 'celebrities', 'reviews', 'pop-culture'],  # Entertainment
        4: ['research', 'discovery', 'space', 'biology', 'physics'],  # Science
        5: ['health', 'wellness', 'travel', 'food', 'personal-growth']  # Lifestyle
    }
    num_tags = np.random.randint(2, 5)
    return np.random.choice(tag_collections[category_id], num_tags, replace=False).tolist()

# Function to generate metadata
def generate_comprehensive_metadata(interactions_df, num_categories=5, category_map=None):
    if category_map is None:
        category_map = {
            1: 'Technology',
            2: 'Sports',
            3: 'Entertainment',
            4: 'Science',
            5: 'Lifestyle'
        }

    # Ensure unique post_ids
    unique_posts = interactions_df['post_id'].unique()
    if len(unique_posts) == 0:
        raise ValueError("No unique post_ids found in interactions_df.")

    # Assign categories and generate metadata
    categories = np.random.randint(1, num_categories + 1, size=len(unique_posts))
    metadata = pd.DataFrame({
        'post_id': unique_posts,
        'category_id': categories,
        'tags': [_generate_tags(cat_id) for cat_id in categories],
        'creator': [f"Creator_{np.random.randint(1, 100)}" for _ in unique_posts],
        'content_type': np.random.choice(['article', 'video', 'blog', 'podcast'], len(unique_posts)),
        'difficulty_level': np.random.choice(['beginner', 'intermediate', 'advanced'], len(unique_posts)),
        'language': np.random.choice(['English', 'Spanish', 'French', 'German'], len(unique_posts)),
        # Corrected random timestamp generation
        'created_at': pd.to_datetime(
            np.random.randint(
                int(pd.Timestamp('2022-01-01').timestamp()),  # Start timestamp in seconds
                int(pd.Timestamp('2024-01-01').timestamp()),  # End timestamp in seconds
                size=len(unique_posts)
            ),
            unit='s'  # Convert seconds to datetime
        )
    })

    metadata['category_name'] = metadata['category_id'].map(category_map)
    return metadata

# Pydantic model for the input data (interaction data)
class InteractionData(BaseModel):
    post_id: int
    viewed_at: Optional[str]
    liked_at: Optional[str]
    inspired_at: Optional[str]
    rated_at: Optional[str]
    rating_percent: Optional[int]

# Example endpoint to generate metadata based on interaction data
@app.post("/generate_metadata/")
def generate_metadata(data: list[InteractionData]):
    try:
        # Convert the input data into a pandas DataFrame
        interaction_data = pd.DataFrame([item.dict() for item in data])

        # Generate the comprehensive metadata
        metadata = generate_comprehensive_metadata(interaction_data)

        # Return the metadata as a JSON response
        return {"metadata": metadata.to_dict(orient="records")}

    except Exception as e:
        return {"error": str(e)}

# Example endpoint to test API
@app.get("/")
def home():
    return {"message": "API is running!"}


# In[ ]:





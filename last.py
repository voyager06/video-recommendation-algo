#!/usr/bin/env python
# coding: utf-8

# In[26]:


import requests


API_ENDPOINTS = {
    "viewed": "https://api.socialverseapp.com/posts/view",
    "liked": "https://api.socialverseapp.com/posts/like",
    "inspired": "https://api.socialverseapp.com/posts/inspire",
    "rated": "https://api.socialverseapp.com/posts/rating"
}


HEADERS = {
    "Flic-Token": "flic_6e2d8d25dc29a4ddd382c2383a903cf4a688d1a117f6eb43b35a1e7fadbb84b8"
}


def fetch_data(api_name):
    """
    Fetch data from the given API name (paginated).
    
    Parameters:
    - api_name: The key in API_ENDPOINTS corresponding to the desired API endpoint.
    
    Returns:
    - all_data: A list of all data retrieved from the API.
    """
    url = API_ENDPOINTS[api_name]
    page = 1
    page_size = 1000
    all_data = []

    while True:
        params = {
            "page": page,
            "page_size": page_size,
            "resonance_algorithm": "resonance_algorithm_cjsvervb7dbhss8bdrj89s44jfjdbsjd0xnjkbvuire8zcjwerui3njfbvsujc5if"
        }
        headers = {"Flic-Token": "flic_6e2d8d25dc29a4ddd382c2383a903cf4a688d1a117f6eb43b35a1e7fadbb84b8"}

        # Make the API request
        response = requests.get(url, params=params, headers=headers)

        if response.status_code == 200:
            
            data = response.json().get("posts", [])
            if not data:
                break  
            all_data.extend(data)
            page += 1  
        else:
            raise Exception(f"Failed to fetch data from {url}: {response.status_code}, {response.text}")

    return all_data





viewed_posts = fetch_data("viewed")
liked_posts = fetch_data("liked")
inspired_posts = fetch_data("inspired")
rated_posts = fetch_data("rated")


print(f"Total Viewed Posts: {len(viewed_posts)}")
print(f"Total Liked Posts: {len(liked_posts)}")
print(f"Total Inspired Posts: {len(inspired_posts)}")
print(f"Total Rated Posts: {len(rated_posts)}")


# In[27]:


import pandas as pd

# Add interaction_type to each dataset
for post in viewed_posts:
    post["interaction_type"] = "viewed"
for post in liked_posts:
    post["interaction_type"] = "liked"
for post in inspired_posts:
    post["interaction_type"] = "inspired"
for post in rated_posts:
    post["interaction_type"] = "rated"

# Convert each list into a DataFrame
viewed_df = pd.DataFrame(viewed_posts)
liked_df = pd.DataFrame(liked_posts)
inspired_df = pd.DataFrame(inspired_posts)
rated_df = pd.DataFrame(rated_posts)


all_data = pd.concat([viewed_df, liked_df, inspired_df, rated_df], ignore_index=True)


print(all_data.head())


# In[28]:


import pandas as pd
import numpy as np
from datetime import datetime



# Step 1: Fill Missing Data
def preprocess_interactions(df):
    df = df.copy()
    timestamp_columns = ['viewed_at', 'liked_at', 'inspired_at', 'rated_at']
    
    
    for col in timestamp_columns:
        if col in df.columns:
            df[col] = df[col].fillna("Not Performed")
        else:
            df[col] = "Not Performed"
    
    
    if 'rating_percent' in df.columns:
        df['rating_percent'] = df['rating_percent'].fillna(0)
    else:
        df['rating_percent'] = 0

    return df


# Step 2: Feature Engineering
def add_interaction_features(df):
    df = df.copy()
    timestamp_columns = ['viewed_at', 'liked_at', 'inspired_at', 'rated_at']
    
    
    for col in timestamp_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    
    current_time = datetime.now()
    if 'viewed_at' in df.columns:
        df['recency_days'] = (current_time - df['viewed_at']).dt.total_seconds() / (24 * 3600)
    else:
        df['recency_days'] = np.nan  
    
    df['recency_days'].fillna(30, inplace=True)  

    
    weights = {'viewed': 1, 'liked': 2, 'inspired': 3}

    
    df['interaction_score'] = df.apply(
        lambda row: (
            weights.get(row['interaction_type'], 0)  
            if row['interaction_type'] != 'rated'  
            else row['rating_percent'] / 100  
        ),
        axis=1
    )

    return df


# Step 3: Aggregate Data
def aggregate_interactions(df):
    return df.groupby(['user_id', 'post_id']).agg(
        total_interaction_score=('interaction_score', 'sum'),
        total_interaction_count=('interaction_type', 'count'),
        most_recent_interaction_days=('recency_days', 'min')
    ).reset_index()


# Step 4: Normalize Scores
def normalize_scores(df):
    df = df.copy()
    score_range = df['total_interaction_score'].max() - df['total_interaction_score'].min()
    
    
    if score_range == 0:
        df['normalized_interaction_score'] = 0
    else:
        df['normalized_interaction_score'] = (
            (df['total_interaction_score'] - df['total_interaction_score'].min()) / score_range
        )

    return df


# Execute the pipeline
processed_data = preprocess_interactions(all_data)  # Step 1
featured_data = add_interaction_features(processed_data)  # Step 2
aggregated_data = aggregate_interactions(featured_data)  # Step 3
final_data = normalize_scores(aggregated_data)  # Step 4


print(final_data.head())


# In[29]:


import pandas as pd


viewed_df = pd.DataFrame(viewed_posts)
liked_df = pd.DataFrame(liked_posts)
inspired_df = pd.DataFrame(inspired_posts)
rated_df = pd.DataFrame(rated_posts)


viewed_df["interaction_type"] = "viewed"
liked_df["interaction_type"] = "liked"
inspired_df["interaction_type"] = "inspired"
rated_df["interaction_type"] = "rated"


all_data = pd.concat([viewed_df, liked_df, inspired_df, rated_df], ignore_index=True)


print(all_data.head())


# In[ ]:





# In[30]:





# In[31]:







# In[32]:


import pandas as pd
import numpy as np
from fastapi import FastAPI
from typing import Optional
from pydantic import BaseModel

# Initialize the FastAPI app
app = FastAPI()


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


def generate_comprehensive_metadata(interactions_df, num_categories=5, category_map=None):
    if category_map is None:
        category_map = {
            1: 'Technology',
            2: 'Sports',
            3: 'Entertainment',
            4: 'Science',
            5: 'Lifestyle'
        }

    
    unique_posts = interactions_df['post_id'].unique()
    if len(unique_posts) == 0:
        raise ValueError("No unique post_ids found in interactions_df.")

    
    categories = np.random.randint(1, num_categories + 1, size=len(unique_posts))
    metadata = pd.DataFrame({
        'post_id': unique_posts,
        'category_id': categories,
        'tags': [_generate_tags(cat_id) for cat_id in categories],
        'creator': [f"Creator_{np.random.randint(1, 100)}" for _ in unique_posts],
        'content_type': np.random.choice(['article', 'video', 'blog', 'podcast'], len(unique_posts)),
        'difficulty_level': np.random.choice(['beginner', 'intermediate', 'advanced'], len(unique_posts)),
        'language': np.random.choice(['English', 'Spanish', 'French', 'German'], len(unique_posts)),
        
        'created_at': pd.to_datetime(
            np.random.randint(
                int(pd.Timestamp('2022-01-01').timestamp()),  # Start timestamp in seconds
                int(pd.Timestamp('2024-01-01').timestamp()),  # End timestamp in seconds
                size=len(unique_posts)
            ),
            unit='s'  
        )
    })

    metadata['category_name'] = metadata['category_id'].map(category_map)
    return metadata


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
        
        interaction_data = pd.DataFrame([item.dict() for item in data])

        
        metadata = generate_comprehensive_metadata(interaction_data)

       
        return {"metadata": metadata.to_dict(orient="records")}

    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def home():
    return {"message": "API is running!"}


# In[33]:


import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class RecommendationEngine:
    def __init__(self, interaction_data, post_metadata):
        """
        Initialize the recommendation engine with robust error handling
        
        Parameters:
        - interaction_data: DataFrame with user-post interactions
        - post_metadata: DataFrame with post details (category, tags, etc.)
        """
        
        self.interaction_data = interaction_data.copy()
        self.post_metadata = post_metadata
        
        
        self._validate_and_preprocess_data()
        
        
        self.user_interaction_profile = self._create_user_interaction_profile()
    
    def _validate_and_preprocess_data(self):
        """
        Validate and preprocess input data with robust fallback mechanisms
        """
        # Define the columns we need
        required_columns = ['user_id', 'post_id']
        optional_columns = [
            'total_interaction_score', 
            'normalized_interaction_score', 
            'recency_days', 
            'viewed_at', 
            'viewed_count', 
            'liked_count', 
            'inspired_count'
        ]
        
        
        for col in required_columns:
            if col not in self.interaction_data.columns:
                raise ValueError(f"Required column '{col}' is missing from the interaction data.")
        
        
        
        
        if 'total_interaction_score' not in self.interaction_data.columns:
            
            if 'viewed_count' in self.interaction_data.columns:
                self.interaction_data['total_interaction_score'] = self.interaction_data['viewed_count']
            else:
               
                self.interaction_data['total_interaction_score'] = 1
        
       
        if 'normalized_interaction_score' not in self.interaction_data.columns:
            try:
                self.interaction_data['normalized_interaction_score'] = (
                    self.interaction_data['total_interaction_score'] - 
                    self.interaction_data['total_interaction_score'].min()
                ) / (
                    self.interaction_data['total_interaction_score'].max() - 
                    self.interaction_data['total_interaction_score'].min()
                )
            except Exception:
                
                self.interaction_data['normalized_interaction_score'] = 0.5
        
        
        if 'recency_days' not in self.interaction_data.columns:
            if 'viewed_at' in self.interaction_data.columns:
                self.interaction_data['recency_days'] = (
                    datetime.now() - pd.to_datetime(self.interaction_data['viewed_at'])
                ).dt.total_seconds() / (24 * 3600)
            else:
                
                self.interaction_data['recency_days'] = 30
    
    def _create_user_interaction_profile(self):
        """
        Create a profile of user interactions with robust aggregation
        """
        
        agg_methods = {
            'post_id': list,
            'total_interaction_score': 'sum',
            'viewed_count': 'sum' if 'viewed_count' in self.interaction_data.columns else lambda x: 0,
            'liked_count': 'sum' if 'liked_count' in self.interaction_data.columns else lambda x: 0,
            'inspired_count': 'sum' if 'inspired_count' in self.interaction_data.columns else lambda x: 0
        }
        
        
        clean_agg_methods = {
            k: v for k, v in agg_methods.items() 
            if k in self.interaction_data.columns
        }
        
        
        user_profile = self.interaction_data.groupby('user_id').agg(clean_agg_methods).reset_index()
        
        return user_profile

def create_fallback_interaction_data(input_data):
    """
    Create a fallback interaction dataset with minimal required information
    
    Parameters:
    - input_data: Original DataFrame to derive fallback data from
    
    Returns:
    - A DataFrame with essential recommendation engine columns
    """
    fallback_data = input_data.copy()
    
    
    if 'user_id' not in fallback_data.columns:
        raise ValueError("No user identifier found in the input data")
    
    if 'post_id' not in fallback_data.columns:
        raise ValueError("No post identifier found in the input data")
    
    
    fallback_data['total_interaction_score'] = 1
    
    
    fallback_data['normalized_interaction_score'] = 0.5
    
    
    fallback_data['recency_days'] = 30
    
    return fallback_data

def initialize_recommendation_engine(interaction_data, post_metadata):
    """
    Safely initialize recommendation engine with fallback mechanisms
    
    Parameters:
    - interaction_data: DataFrame with user-post interactions
    - post_metadata: DataFrame with post details
    
    Returns:
    - Initialized RecommendationEngine instance
    """
    try:
        # First attempt: Use original data
        recommendation_engine = RecommendationEngine(
            interaction_data=interaction_data,
            post_metadata=post_metadata
        )
        return recommendation_engine
    
    except Exception as e:
        print(f"Error initializing with original data: {e}")
        print("Attempting to create fallback interaction data...")
        
        try:
            # Second attempt: Use fallback data creation
            fallback_interaction_data = create_fallback_interaction_data(interaction_data)
            recommendation_engine = RecommendationEngine(
                interaction_data=fallback_interaction_data,
                post_metadata=post_metadata
            )
            return recommendation_engine
        
        except Exception as fallback_error:
            print(f"Fallback data creation failed: {fallback_error}")
            raise ValueError("Unable to initialize recommendation engine with given data")


# In[34]:


from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

def recommend_content_based(user_id, interaction_data, post_metadata):
    
    user_posts = interaction_data[interaction_data['user_id'] == user_id]['post_id']
    user_history_metadata = post_metadata[post_metadata['post_id'].isin(user_posts)]
    
    
    post_metadata['combined_features'] = post_metadata['tags'] + ' ' + post_metadata['category']
    
    
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(post_metadata['combined_features'])
    
    
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    
    user_indices = post_metadata[post_metadata['post_id'].isin(user_posts)].index
    similar_posts = similarity_matrix[user_indices].sum(axis=0).argsort()[::-1]
    recommendations = post_metadata.iloc[similar_posts][:10]  # Top 10
    
    return recommendations


# In[35]:


from sklearn.neighbors import NearestNeighbors

def recommend_collaborative(user_id, interaction_data):
    
    user_post_matrix = interaction_data.pivot_table(
        index='user_id', columns='post_id', values='normalized_interaction_score', fill_value=0
    )
    
    
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(user_post_matrix)
    
    
    user_vector = user_post_matrix.loc[user_id].values.reshape(1, -1)
    distances, indices = knn.kneighbors(user_vector, n_neighbors=5)  # Top 5 neighbors
    
    
    similar_users = user_post_matrix.iloc[indices.flatten()].index
    recommendations = interaction_data[
        interaction_data['user_id'].isin(similar_users)
    ]['post_id'].unique()
    
    return recommendations


# In[36]:


def rank_posts(posts, interaction_data):
    posts['final_score'] = (
        posts['normalized_interaction_score'] * (1 / (1 + posts['recency_days']))
    )
    return posts.sort_values(by='final_score', ascending=False).head(10)


# In[ ]:





# In[37]:


def recommend_by_category(category_id, post_metadata, interaction_data):
    category_posts = post_metadata[post_metadata['category_id'] == category_id]
    ranked_posts = rank_posts(category_posts, interaction_data)
    return ranked_posts.head(10)


# In[38]:


mood_to_tags = {
    "happy": ["comedy", "funny", "uplifting"],
    "relaxed": ["nature", "calm", "meditation"]
}

def recommend_by_mood(mood, post_metadata, interaction_data):
    tags = mood_to_tags.get(mood, [])
    mood_posts = post_metadata[post_metadata['tags'].apply(lambda x: any(tag in x for tag in tags))]
    ranked_posts = rank_posts(mood_posts, interaction_data)
    return ranked_posts.head(10)


# In[39]:


def recommend_trending(post_metadata, interaction_data):
    return rank_posts(post_metadata, interaction_data).head(10)


# In[40]:


def recommend_posts(user_id=None, category_id=None, mood=None, interaction_data=None, post_metadata=None):
    if user_id and user_id in interaction_data['user_id'].unique():  
        personalized_recommendations = recommend_content_based(user_id, interaction_data, post_metadata)
        return personalized_recommendations
    else:  
        if category_id:
            return recommend_by_category(category_id, post_metadata, interaction_data)
        elif mood:
            return recommend_by_mood(mood, post_metadata, interaction_data)
        else:
            return recommend_trending(post_metadata, interaction_data)


# In[41]:


def precision_recall_at_k(recommendations, ground_truth, k):
    """
    Compute Precision@K and Recall@K.
    
    Parameters:
    - recommendations: List of recommended post_ids
    - ground_truth: List of actual post_ids the user interacted with
    - k: Number of top recommendations to evaluate
    
    Returns:
    - precision: Proportion of relevant items in top-K recommendations
    - recall: Proportion of relevant items retrieved
    """
    recommended_top_k = recommendations[:k]
    relevant_items = set(ground_truth)
    
    
    retrieved_relevant = set(recommended_top_k).intersection(relevant_items)
    
    
    precision = len(retrieved_relevant) / k
    recall = len(retrieved_relevant) / len(relevant_items) if relevant_items else 0
    
    return precision, recall


# In[42]:


import numpy as np

def ndcg_at_k(recommendations, ground_truth, k):
    """
    Compute NDCG@K.
    
    Parameters:
    - recommendations: List of recommended post_ids
    - ground_truth: List of actual post_ids the user interacted with
    - k: Number of top recommendations to evaluate
    
    Returns:
    - NDCG@K score
    """
    recommended_top_k = recommendations[:k]
    ideal_rank = sorted(ground_truth, reverse=True)[:k]  
    
    
    dcg = sum([1 / np.log2(idx + 2) for idx, rec in enumerate(recommended_top_k) if rec in ground_truth])
    
    idcg = sum([1 / np.log2(idx + 2) for idx, _ in enumerate(ideal_rank)])
    
    return dcg / idcg if idcg > 0 else 0


# In[43]:


#pip install fastapi uvicorn


# In[45]:


import requests
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from typing import Optional
import uvicorn


app = FastAPI(
    title="Content Recommendation API",
    description="API for generating personalized content recommendations",
    version="1.0.0"
)


INTERACTION_DATA_API = "https://api.socialverseapp.com/posts/view"
POST_METADATA_API = "http://127.0.0.1:8000/generate_metadata/"

def fetch_mock_data():
    """
    Generate mock data for development and testing
    """
    
    interaction_data = pd.DataFrame({
        'user_id': [1, 1, 2, 2, 3],
        'post_id': [101, 102, 201, 202, 301],
        'viewed_at': pd.date_range(start='2024-01-01', periods=5),
        'rating_percent': [80, 90, 70, 60, 75]
    })

    
    metadata = pd.DataFrame({
        'post_id': [101, 102, 201, 202, 301],
        'category_id': [1, 2, 3, 4, 5],
        'category_name': ['Technology', 'Sports', 'Entertainment', 'Science', 'Lifestyle'],
        'tags': [
            ['innovation', 'tech'],
            ['fitness', 'football'],
            ['movies', 'celebrities'],
            ['research', 'space'],
            ['health', 'wellness']
        ],
        'created_at': pd.date_range(start='2024-01-01', periods=5)
    })

    return interaction_data, metadata

def fetch_interaction_data():
    """
    Fetch interaction data, with fallback to mock data
    """
    try:
        response = requests.get(INTERACTION_DATA_API)
        if response.status_code == 200:
            return pd.DataFrame(response.json()["posts"])
    except Exception as e:
        print(f"Error fetching interaction data: {e}")
    
    
    interaction_data, _ = fetch_mock_data()
    return interaction_data

def fetch_post_metadata():
    """
    Fetch post metadata, with fallback to mock data
    """
    try:
        response = requests.get(POST_METADATA_API)
        if response.status_code == 200:
            return pd.DataFrame(response.json()["metadata"])
    except Exception as e:
        print(f"Error fetching post metadata: {e}")
    
    
    _, metadata = fetch_mock_data()
    return metadata

@app.get("/")
def home():
    return {
        "message": "Welcome to the Content Recommendation API",
        "available_endpoints": [
            "/recommend",
            "/feed"
        ]
    }

@app.get("/recommend")
def recommend(
    user_id: Optional[int] = Query(None, description="User ID for personalized recommendations"),
    category_id: Optional[int] = Query(None, description="Category ID to filter recommendations"),
    mood: Optional[str] = Query(None, description="Mood to filter recommendations"),
    top_k: int = Query(10, description="Number of recommendations to return")
):
    """
    Recommend posts based on user_id, category_id, or mood.
    """
    try:
        
        interaction_data = fetch_interaction_data()
        post_metadata = fetch_post_metadata()

        
        if user_id:
            user_posts = interaction_data[interaction_data['user_id'] == user_id]['post_id']
            recommendations = post_metadata[post_metadata['post_id'].isin(user_posts)]
        elif category_id:
            recommendations = post_metadata[post_metadata['category_id'] == category_id]
        else:
            recommendations = post_metadata

        
        recommendations = recommendations.sort_values('created_at', ascending=False)
        return {"recommendations": recommendations.head(top_k).to_dict(orient="records")}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# In[ ]:





# In[ ]:





# In[ ]:





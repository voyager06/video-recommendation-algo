# Content Recommendation System

## Project Overview

This is a sophisticated recommendation system built using Python, FastAPI, and advanced machine learning techniques. The system provides personalized content recommendations based on user interactions, content metadata, and various recommendation strategies.

## System Architecture

### Key Components
1. Data Fetching
2. Interaction Processing
3. Recommendation Engines
4. API Endpoints
5. Evaluation Metrics

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Required Dependencies
```bash
pip install fastapi
pip install uvicorn
pip install requests
pip install pandas
pip install scikit-learn
pip install numpy
```

## Project Structure

### Main Modules

#### 1. Data Fetching (`mainn.py`)
- Retrieves user interaction data from multiple API endpoints
- Supports paginated data retrieval
- Handles different types of interactions (views, likes, inspiration, ratings)

#### 2. Data Preprocessing
- Normalizes interaction data
- Calculates interaction scores
- Creates user interaction profiles
- Handles missing data gracefully

#### 3. Recommendation Strategies

##### Content-Based Recommendation
- Uses TF-IDF vectorization
- Computes content similarity
- Recommends posts based on user's historical content preferences

##### Collaborative Filtering
- Employs K-Nearest Neighbors algorithm
- Finds similar users
- Recommends posts liked by similar users

##### Additional Recommendation Methods
- Category-based recommendations
- Mood-based recommendations
- Trending content recommendations

#### 4. Evaluation Metrics
- Precision@K
- Recall@K
- Normalized Discounted Cumulative Gain (NDCG)

## API Endpoints

### 1. Home Endpoint (`/`)
- Returns basic API information
- Provides available endpoints

### 2. Recommendation Endpoint (`/recommend`)
**Parameters:**
- `user_id`: Optional user identifier
- `category_id`: Optional category filter
- `mood`: Optional mood-based filter
- `top_k`: Number of recommendations (default: 10)

**Response:**
- JSON array of recommended posts

## Recommendation Logic Flow

1. Data Retrieval
   - Fetch user interaction data
   - Fetch post metadata

2. Recommendation Generation
   - For existing users: Content-based recommendations
   - For new users: 
     * Category-based recommendations
     * Mood-based recommendations
     * Trending content

3. Ranking
   - Score posts based on:
     * Interaction score
     * Recency
     * Relevance

## Advanced Features

### 1. Fallback Mechanisms
- Mock data generation
- Error handling for API calls
- Graceful degradation

### 2. Flexible Configuration
- Easily configurable recommendation weights
- Extensible recommendation strategies

## Running the Application

### Development Mode
```bash
uvicorn main:app --reload
```

### Production Deployment
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Performance Evaluation

### Metrics Tracked
- Precision
- Recall
- Normalized Discounted Cumulative Gain (NDCG)

## Potential Improvements
- Add more sophisticated machine learning models
- Implement caching mechanisms
- Enhance mood and category detection
- Real-time recommendation updates

## Troubleshooting

### Common Issues
1. API Endpoint Unavailability
   - Check network connectivity
   - Verify API credentials
   - Use mock data for testing

2. Performance Bottlenecks
   - Monitor response times
   - Implement result caching
   - Optimize recommendation algorithms

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request



## Contact
shares.zoro@gmail.com
https://www.linkedin.com/in/vedant-swami-768162245/

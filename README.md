# Spotify Genre Prediction

## i. The Spotify Dataset
For this project, we used a dataset derived from HuggingFace, containing a tabular dataset of Spotify tracks in CSV format with 114,000 rows and 20 columns. It was collected using the Spotify Web API and Python. The unit of observation is one Spotify track (or song). The dataset includes the following features:

- **track_id**: The Spotify ID for the track.
- **artists**: The artists' names who performed the track.
- **album_name**: The album name in which the track appears.
- **track_name**: Name of the track.
- **popularity**: A value between 0 and 100, with 100 being the most popular.
- **duration_ms**: The track length in milliseconds.
- **explicit**: Whether or not the track has explicit lyrics.
- **danceability**: Suitability of a track for dancing. A value of 0.0 is least danceable and 1.0 is most danceable.
- **energy**: A measure from 0.0 to 1.0 representing the perceptual intensity and activity.
- **key**: The key the track is in.
- **loudness**: The overall loudness of a track in decibels (dB).
- **mode**: Modality (major or minor) of a track.
- **speechiness**: The presence of spoken words in a track. The closer to 1.0, the more speech-like the recording.
- **acousticness**: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence.
- **instrumentalness**: Predicts whether a track contains no vocals. The closer the value is to 1.0, the greater likelihood the track contains no vocal content.
- **liveness**: Detects the presence of an audience in the recording.
- **valence**: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track.
- **tempo**: The overall estimated tempo of a track in beats per minute (BPM).
- **time_signature**: An estimated time signature.
- **track_genre**: The genre to which the track belongs.

The target variable is **track_genre**, representing the track's genre.

## ii. Overview of the Problem
The focus of this project is to classify the genre of a track based on its feature attributes and ultimately predict the genre of a new track given its feature attributes. The dataset includes a total of 114 unique genres at relatively balanced counts. To reduce the complexity of the problem and enable accurate predictions, we narrowed the dataset to tracks belonging to twenty genres, selected from the most populated classes after preprocessing.

## iii. Key Methodology

1. **Data Preprocessing**:
   - Removed null values, duplicate rows, and irrelevant features (e.g., artist names, track IDs).
   - Performed feature selection using LASSO logistic regression and random forests to identify the top 10 most impactful features for genre classification.

2. **Model Training**:
   - Trained a random forest classifier, achieving an accuracy of 68%. Random forests aggregate predictions of multiple decision trees trained on subsets of the data, reducing overfitting and improving generalizability.

## iv. Results

| Model                  | Test Accuracy | Validation Accuracy |
|------------------------|---------------|---------------------|
| Logistic Regression    | 55.0%         | 53.1%               |
| Random Forests         | 65.0%         | 64.1%               |
| Decision Trees         | 55.2%         | 50.1%               |
| 4-Layer Neural Network | 57.7%         | 57.4%               |
| 6-Layer Neural Network | 61.0%         | 60.4%               |
| 10-Layer Neural Network| 59.1%         | 51.9%               |

Random Forests yielded the highest test and validation accuracy among the six models evaluated. We employed 5-fold cross-validation to ensure robust performance comparisons, mitigating overfitting and providing reliable performance estimates on unseen data.

### Model Insights:
- **Random Forests**:
  - Excelled due to their ability to handle high-dimensional data and robustness to noise.
  - Inclusion of feature randomness reduced overfitting compared to simpler models like decision trees.

- **Logistic Regression**:
  - Assumes linear relationships, limiting performance on datasets with complex interactions.

- **Neural Networks**:
  - Require large datasets and extensive tuning. The dataset size may have been insufficient to leverage their full potential.

### Future Improvements:
- Implement dimensionality reduction techniques or feature engineering to simplify the feature space and reduce noise.
- Explore complementary methods to refine predictive capabilities further.


# DCC-Capstone

## DataSets

- [Genius song lyrics~5mil](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information?select=song_lyrics.csv)
- [Full TMDB Movies Dataset 2024~1mil](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies)

## Key Libraries

This Project uses:
    - pandas for data cleaning

    - re for text preprocessing

    - torch for CUDA and neural net architecture

    - transformers for embeddings
    
    - sklearn for Machine learning and similarity based recommendations


## Overview: 

This project uses only uses songs and movies in english. Songs are sampled evenly based off of the tag distribution in order to ensure an even distribution of all genres. Cleaning can be found in the 3M_songs_filtering.ipynb and movies_filtering_embedding.ipynb files. 

Embedding on song lyrics and movie overviews are done using the hugging face [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) model from the sentence-transformers library. the embedding process can be found in the songs_embedding.ipynb and movies_filtering_embedding.ipynb files.

This project trains an adversarial autoencoder to put the song embeddings and the movie embeddings into the same feature space. The training architecture for the model can be found in AdversarialAutoencoder.ipynb. The model used can be found in best_aae_model.pt. Use of the model can be seen in AutoencoderBasedRecomender.ipynb or in recommendation_calculation.ipynb

The project has two recommendation systems, one based off of the embeddings obtained with the [sentence-transformers/all-mpnet-base-v2](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) model, and one based off of the embeddings obtained from the best_aae_model. A sample recommendation for five songs for the aae based embeddings can be found in the AutoencoderBasedRecomender.ipynb file. In addition a recommendation for a single song of your choosing for the base model can be found in the single_song_recommendation.ipynb file. The single song recommendation can only take in songs found in the [Genius song lyrics](https://www.kaggle.com/datasets/carlosgdcj/genius-song-lyrics-with-language-information?select=song_lyrics.csv) dataset. Recomendations for the whole dataset for both models can be found in recommendation_calculation.ipynb.

The project evaluates a random sample of the recommendations using [gemini-2.5-flash-preview-04-17](https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini/2-5-flash). 

## Important

The [Full TMDB Movies Dataset 2024](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies) contains adult movies. If you'd like to exclude adult movies, you can drop the movies where the adult column in true. As is the model does not remove adult movies. 
import time
import json
import pickle
import os, re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from datetime import timedelta
from IPython.display import clear_output
from nltk.tokenize.regexp import RegexpTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

#load our huggingface information
with open('../langchain/keys/hf.json', 'r') as rd_f:
    data = json.load(rd_f)
    
os.environ['HUGGINGFACEHUB_API_TOKEN'] = data['HUGGINGFACEHUB_API_TOKEN']

class MovieRecommender:
    def __init__(self, path='./'):
        #load variables needed for searching
        self.movie_dict = self.load_data()
        self.model = self.load_embedding_model()
        self.tokenizer = RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
        
        #load semantic database or create new database if one is not found
        try:
            self.embeddings = self.load_vector_store(path)
            self.tfidf, self.tfidf_embeddings = self.load_vectorizer(path)
        except:
            #start timer
            start = time.perf_counter()
            print(f"Creating database for {len(self.movie_dict)} movies...")
            
            #create database
            self.create_vector_store(path)
            #create tfidf database
            self.create_and_save_tfidf(self.movie_dict, path)
            clear_output()
            print(f"Time to create database: {timedelta(seconds=time.perf_counter() - start)}")
            
            #save data to new folder
            self.save_data(path)
            #load database
            self.embeddings = self.load_vector_store(path)
            #load tfidf database
            self.tfidf, self.tfidf_embeddings = self.load_vectorizer(path)
            
    def load_data(self, df_path='./movies_database.json'):
        """
        Load The Movie Database json file of movies.
        """
        with open(df_path) as rd_f:
            movies = json.load(rd_f)
        return movies
    
    def save_data(self, path):
        """
        Save json file to folder with all embeddings.
        """
        with open(f"{path}/MovieDatabase/movies_database.json", 'w') as wt_f:
            json.dump(self.movie_dict, wt_f)
    
    def load_embedding_model(self):
        """
        Load model for embedding movie overview.
        """
        #load model from sentence transformers into HuggingFaceEmbeddings
        model = SentenceTransformer('all-mpnet-base-v2')
        return model
    
    def create_vector_store(self, path):
        """
        Create semantic database of movie overview embeddings.
        """
        #load and clean data
        data = [movie['overview'] for movie in self.movie_dict]
        
        #create sentence embeddings
        embeddings = self.model.encode(data, show_progress_bar=True)
        
        #create directory for embeddings
        try:
            os.mkdir(f"{path}/MovieDatabase")
        except:
            pass
        #save numpy array
        np.save(f"{path}/MovieDatabase/embeddings", embeddings)
        
    def load_vector_store(self, path):
        """
        Load semantic embeddings.
        """
        #load vector store
        embeddings = np.load(f"{path}/MovieDatabase/embeddings.npy")
        return embeddings
    
    def preprocess_text(self, text):
        """
        Clean text before tfidf.
        """
        text = text.lower()
        tokens = self.tokenizer.tokenize(text)
        tokens = [word for word in tokens if word not in ENGLISH_STOP_WORDS]
        return ' '.join(tokens)
    
    def create_and_save_tfidf(self, movies, path):
        """
        Create tfidf embeddings.
        """
        text_list = [movie['overview'] for movie in movies]
        #preprocess text
        preprocessed_movies = [self.preprocess_text(text) for text in text_list]
        #instantiate tfidf
        tfidf = TfidfVectorizer(analyzer='word')
        #vectorize data
        movie_tfidf = tfidf.fit(preprocessed_movies)
        #fit to data
        movie_vectors = tfidf.transform(preprocessed_movies)

        #create folder to store information
        if not os.path.exists(path):
            os.mkdir(path)
        #save vectorizer
        pickle.dump(movie_tfidf, open(f"./{path}/MovieDatabase/movie_vectorizer.pickle", "wb"))
        #save vectorized data
        pickle.dump(movie_vectors, open(f"./{path}/MovieDatabase/movie_tfidf.pickle", "wb"))
        
            
    def load_vectorizer(self, path):
        """
        Load tfidf embeddings and tfidf vectorizer.
        """
        #load vectorizer
        vectorizer = pickle.load(open(f"./{path}/MovieDatabase/movie_vectorizer.pickle", "rb"))
        #load vectorized data
        movies = pickle.load(open(f"./{path}/MovieDatabase/movie_tfidf.pickle", "rb"))
        return vectorizer, movies
    
    def find_search_title(self, search):
        """
        Find highest name similarity and associated semantic and tfidf embeddings.
        """
        max_sim = 0
        match = None
        
        #iterate through all movie names to try and find movie searched in database
        for loc in range(len(self.movie_dict)):
            #fuzzy matching to find highest similarity to search in database
            sim = fuzz.WRatio(search, self.movie_dict[loc]['title'])
            
            #store highest similarity
            if sim > max_sim:
                match = loc
                max_sim = sim
        return self.movie_dict[match]['genres'], self.embeddings[match], self.tfidf_embeddings[match]
    
    def movie_recommendations(self, match, tfidf_match, semantic_ratio, keyword_ratio, 
                              genres=None, name_search=True, remove_movie_names=[]):
        """
        Combine key word and semantic results from cosine similarity to get top matches.
        """
        #skip first row if name was used to search
        start_point = 1 if name_search else 0
        
        #if genre input then find locations of all movies in-genre
        if isinstance(genres, str):
            #locations of rows in specific genre
            genre_data_idx = [i for i in range(len(self.movie_dict))
                              if (self.movie_dict[i]['title'] not in remove_movie_names)
                              and (genres in self.movie_dict[i]['genres'])] 
            
        else:
            #all entries except those removed from list
            genre_data_idx = [i for i in range(len(self.movie_dict))
                              if (self.movie_dict[i]['title'] not in remove_movie_names)]
            
        #create a filtered movie dict since the corpus_id from semantic search will be different
        movie_dict = [self.movie_dict[i] for i in genre_data_idx]

        #cosine similarity between semantic embeddings
        scores = cosine_similarity(match.reshape(1, -1), self.embeddings[genre_data_idx])[0]
        #cosine similarity between key word embedding
        scores_tfidf = tfidf_match.dot(self.tfidf_embeddings[genre_data_idx].transpose()).toarray()[0]
        
        #combine semantic search and keyword search
        total_score = (scores * semantic_ratio) + (scores_tfidf * keyword_ratio)
        
        #sort for highest score
        indexes = total_score.argsort()[::-1][start_point:]

        #return metadata for movies with highest similarity
        recommendations = [movie_dict[i] for i in indexes]
        return recommendations
    
    def search_movies(self, search, num_results, semantic_ratio=0.9, keyword_ratio=0.1,
                      genres=None, remove_movie_names=[]):
        """
        Get movie recommendations using search title or description.
        """
        #if search is less than 8 words then search by name else by description by user
        search_by_name = True if len(search.split()) < 10 else False
        
        #search for movie in database if True
        if search_by_name:
            best_match_genres, search_sem, search_tfidf = self.find_search_title(search)
            num_results += 1
        else:
            search_sem = self.model.encode(search)
            search_tfidf = self.tfidf.transform([self.preprocess_text(search)])
            best_match_genres = []
            
        #search database for similar movies
        movie_dict = self.movie_recommendations(search_sem, search_tfidf, semantic_ratio, keyword_ratio,
                                                genres, search_by_name, remove_movie_names)
        return movie_dict[:num_results], best_match_genres
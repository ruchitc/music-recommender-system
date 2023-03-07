import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.oauth2 import SpotifyOAuth
import spotipy.util as util

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

from decouple import config

import json

import os
from server.settings import BASE_DIR


class Recommender:
    def __init__(self):
        path_to_artifacts = "../../../notebook/"
        # self.path_to_dataset = "D:\programsData\python\projects\music-recommender-system\dataset\SpotifyFeatures.csv"
        # self.path_to_dataset = "..\..\..\..\..\dataset\SpotifyFeatures.csv"
        self.path_to_dataset = os.path.join(BASE_DIR, '..\..\dataset\SpotifyFeatures.csv')

        client_id = config("client_id")
        client_secret = config("client_secret")

        scope = 'user-library-read'

        client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        self.sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)
    
    def loadDataset(self):
        self.playlist_DF = pd.read_csv(self.path_to_dataset)

        # Feature scaling
        playlistDF_num = self.playlist_DF[['popularity', 'acousticness', 'danceability', 'duration_ms', 'energy', 'instrumentalness',	'liveness',	'loudness',	'speechiness',	'tempo',	'valence']]
        playlistDF_text = self.playlist_DF[['genre', 'key', 'mode']]

        # Numerical data scaling
        scaler = MinMaxScaler()
        num_df = pd.DataFrame(scaler.fit_transform(playlistDF_num), columns=playlistDF_num.columns)
        num_df.reset_index(drop = True, inplace = True)

        # Categorical data scaling
        text_df = pd.get_dummies(playlistDF_text, prefix=['genre', 'key', 'mode'])
        text_df.reset_index(drop = True, inplace = True)

        # Concatenating the numerical and categorical data
        self.final_df = pd.concat([num_df, text_df], axis = 1)
        self.final_df['track_id'] = self.playlist_DF['track_id'].values
    
    def loadPlaylist(self, playlist_url):
        playlist_uri = playlist_url.split("/")[-1].split("?")[0]

        playlist = pd.DataFrame()

        for ix, i in enumerate(self.sp.playlist(playlist_uri)['tracks']['items']):
            #print(i['track']['artists'][0]['name'])
            playlist.loc[ix, 'artist'] = i['track']['artists'][0]['name']
            playlist.loc[ix, 'name'] = i['track']['name']
            playlist.loc[ix, 'id'] = i['track']['id'] # ['uri'].split(':')[2]
            playlist.loc[ix, 'url'] = i['track']['album']['images'][1]['url']
            playlist.loc[ix, 'date_added'] = i['added_at']

        playlist['date_added'] = pd.to_datetime(playlist['date_added'])  
            
        playlist = playlist[playlist['id'].isin(self.playlist_DF['track_id'].values)].sort_values('date_added',ascending = False)

        complete_playlist = self.final_df[self.final_df['track_id'].isin(playlist['id'].values)]
        self.complete_not_in_playlist = self.final_df[~self.final_df['track_id'].isin(playlist['id'].values)]
        playlist_final = complete_playlist.drop(columns = "track_id")
        self.complete_playlist_summary_vector = playlist_final.sum(axis = 0)

    def findCosineSimilarity(self):
        non_playlist_df = self.playlist_DF[self.playlist_DF['track_id'].isin(self.complete_not_in_playlist['track_id'].values)]
        non_playlist_df['cosine_sim'] = cosine_similarity(self.complete_not_in_playlist.drop('track_id', axis = 1).values, self.complete_playlist_summary_vector.values.reshape(1, -1))[:,0]
        non_playlist_df_top_20 = non_playlist_df.sort_values('cosine_sim',ascending = False).head(20)

        return non_playlist_df_top_20

    # def postProcessing(self):
    #     pass

    def computeRecommendation(self, json_obj):
        # data = json.loads(json_obj)
        playlist_link = json_obj['link']
        try:
            self.loadDataset()
            self.loadPlaylist(playlist_link)
            playlist = self.findCosineSimilarity()
        except Exception as e:
            return {"status": "Error", "recommendation": "NULL", "message": str(e)}

        return {"status": "OK", "recommendation": playlist}

# driver code
# input_data = "https://open.spotify.com/playlist/37i9dQZF1DXbTxeAdrVG2l?si=233815c496e54e41"
# input_data = "https://open.spotify.com/playlist/2OQTMm3MMqUMDRWJaafxWE?si=e58552b26479475f"

# response = pd.DataFrame()

# recommender = Recommender()
# response = recommender.computeRecommendation(input_data)
# print(response['recommendation'])
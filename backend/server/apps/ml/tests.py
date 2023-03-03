from django.test import TestCase

import pandas as pd

from apps.ml.recommender.recommender import Recommender

import inspect
from apps.ml.registry import MLRegistry

class MLTests(TestCase):
    '''
    def test(self):
        input_data = "https://open.spotify.com/playlist/37i9dQZF1DXbTxeAdrVG2l?si=233815c496e54e41"
        # input_data = "https://open.spotify.com/playlist/2OQTMm3MMqUMDRWJaafxWE?si=e58552b26479475f"

        filePath = "..\..\..\..\dataset\SpotifyFeatures.csv"
        response = pd.DataFrame()

        recommender = Recommender(filePath)
        response = recommender.computeRecommendation(input_data)
        print(response)
        self.assertEqual('OK', response['status'])
    '''

    def test_registry(self):
        registry = MLRegistry()
        self.assertEqual(len(registry.endpoints), 0)

        endpoint_name = "recommender"
        algorithm_object = Recommender()
        algorithm_name = "cosine_similarity"
        algorithm_status = "production"
        algorithm_version = "0.0.1"
        algorithm_owner = "ruchit"
        algorithm_description = "Music recommender using cosine similarity"
        algorithm_code = inspect.getsource(Recommender)

        registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
                               algorithm_status, algorithm_version, algorithm_owner,
                               algorithm_description, algorithm_code)
        
        self.assertEqual(len(registry.endpoints), 1)
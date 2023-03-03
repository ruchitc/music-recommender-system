"""
WSGI config for server project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.1/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'server.settings')

application = get_wsgi_application()

import inspect
from apps.ml.registry import MLRegistry
from apps.ml.recommender.recommender import Recommender

try:
    registry = MLRegistry()
    recommender = Recommender()

    registry.add_algorithm(endpoint_name = "recommender",
                           algorithm_object = recommender,
                           algorithm_name = "cosine_similarity",
                           algorithm_status = "production",
                           algorithm_version = "0.0.1",
                           owner = "ruchit",
                           algorithm_description = "Music recommender using cosine similarity",
                           algorithm_code = inspect.getsource(Recommender))

except Exception as e:
    print("Exception while loading the algorithms to the registry, ", str(e))

# -*- coding: utf-8 -*-

import sys
import traceback

from gensim.models import word2vec
from werkzeug.exceptions import InternalServerError

from utilities.bigquery import get_training_corpus_dataframe
from utilities.object_repository import ObjectRepository, Kind

NUM_WORKERS = 2
NUM_FEATURES = 200
MIN_WORD_COUNT = 5
CONTEXT_SIZE = 3


class RecommendationModel:
    """This is where the ML magic happens"""

    def __init__(self):
        self.repository = ObjectRepository()

    def train(self, content_space_id):
        training_corpus = self._split_data(get_training_corpus_dataframe(content_space_id))
        try:

            embedding = word2vec.Word2Vec(training_corpus, workers=NUM_WORKERS,
                                          size=NUM_FEATURES, min_count=MIN_WORD_COUNT,
                                          window=CONTEXT_SIZE)

        except Exception:
            print(traceback.format_exc(), file=sys.stderr)
            raise InternalServerError(
                description="Error training model for content_space_id '{}'".format(content_space_id)
            )

        self.repository.store_object(content_space_id, Kind.UNIVERSAL, embedding)

    def get_most_similar(self, content_space_id, key):
        embedding = self.repository.get_object(content_space_id, Kind.UNIVERSAL)
        try:

            similar_entities = embedding.wv.most_similar(key)

        except KeyError:
            return []
        except Exception:
            print(traceback.format_exc(), file=sys.stderr)
            raise InternalServerError(
                description="Error using model for content_space_id '{}'".format(content_space_id)
            )

        similar_entities.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in similar_entities]

    def get_recommendation_order(self, content_space_id, key_list):
        """key_list is a list of model, inventory or community keys"""
        embedding = self.repository.get_object(content_space_id, Kind.UNIVERSAL)
        try:

            model_vector = sum([embedding.wv[key] for key in key_list if key in embedding.wv.vocab]) / len(key_list)
            ordered_entities = embedding.wv.similar_by_vector(model_vector, topn=1000)

        except TypeError:
            return []
        except Exception:
            print(traceback.format_exc(), file=sys.stderr)
            raise InternalServerError(
                description="Error using model for content_space_id '{}'".format(content_space_id)
            )

        ordered_entities.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in ordered_entities]

    # -- internal helper functions -- #

    @staticmethod
    def _split_data(corpus_dataframe):
        # split data into model, inventory, community for training corpus
        corpus = corpus_dataframe.groupby("session_id")["id"].apply(list)
        return corpus.to_list()

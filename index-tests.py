import unittest
from ipynb.fs.full.index import (movies, remove_movies_missing_data)
import math
class TestApplyGradientDescentLab(unittest.TestCase):
    def test_remove_nan_values(self):
        unfilteredMovies = list(filter(lambda movie: math.isnan(movie['domgross_2013$']),movies))
        self.assertEqual(len(unfilteredMovies), 18)
        parsed_movies = remove_movies_missing_data(movies)
        nan_parsed_movies = list(filter(lambda movie: math.isnan(movie['domgross_2013$']),parsed_movies))
        self.assertEqual(len(nan_parsed_movies), 0)

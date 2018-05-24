import unittest
import sys
sys.path.insert(0, '..')
from ipynb.fs.full.index import (movies, remove_movies_missing_data, scale_down_movie,
    scale_down_movies, revenues_per_budgets_trace, highest_domestic_gross, outside_consultant_predicted_revenue,
    consultant_estimated_revenues_trace, error_for_consultant_model, scaled_movies, rss_consultant, revenue_with_year, internal_consultant_estimated_trace,
    squared_error_revenue_with_year, expected_revenue_per_budget, rss_revenue_with_year, initial_regression_trace,
    regression_revenue_error, rss_trace, m_gradient, b_gradient, step_gradient)
import math

class TestApplyGradientDescentLab(unittest.TestCase):
    parsed_movies = remove_movies_missing_data(movies)


    def test_remove_movies_missing_data(self):
        unfilteredMovies = list(filter(lambda movie: math.isnan(movie['domgross_2013$']),movies))
        self.assertEqual(len(unfilteredMovies), 18)
        parsed_movies = remove_movies_missing_data(movies)
        nan_parsed_movies = list(filter(lambda movie: math.isnan(movie['domgross_2013$']),parsed_movies))
        self.assertEqual(len(nan_parsed_movies), 0)

    def test_remove_nan_values(self):
        unfilteredMovies = list(filter(lambda movie: math.isnan(movie['domgross_2013$']),movies))
        self.assertEqual(len(unfilteredMovies), 18)
        parsed_movies = remove_movies_missing_data(movies)
        nan_parsed_movies = list(filter(lambda movie: math.isnan(movie['domgross_2013$']),parsed_movies))
        self.assertEqual(len(nan_parsed_movies), 0)

    def test_scale_down_movie(self):
        first_movie = {'binary': 'FAIL', 'budget': 13000000,
        'budget_2013$': 13000000, 'clean_test': 'notalk',
        'code': '2013FAIL', 'decade code': 1.0, 'domgross': 25682380.0,
        'domgross_2013$': 25682380.0, 'imdb': 'tt1711425', 'intgross': 42195766.0,
        'intgross_2013$': 42195766.0, 'period code': 1.0, 'test': 'notalk',
        'title': '21 &amp; Over', 'year': 2013}
        scaled_movie = {'binary': 'FAIL', 'budget': 13.0,
        'budget_2013$': 13.0, 'clean_test': 'notalk',
        'code': '2013FAIL', 'decade code': 1.0, 'domgross': 25.68,
        'domgross_2013$': 25.68, 'imdb': 'tt1711425', 'intgross': 42.2,
        'intgross_2013$': 42.2, 'period code': 1.0, 'test': 'notalk',
        'title': '21 &amp; Over', 'year': 2013}
        self.assertEqual(scale_down_movie(first_movie), scaled_movie)

    def test_scale_down_movies(self):
        parsed_movies = self.parsed_movies
        first_ten_movies = parsed_movies[0:10]
        scaled_down = scale_down_movies(first_ten_movies)[-2:]
        scaled_nine_and_ten = [{'binary': 'PASS', 'budget': 13.0, 'budget_2013$': 13.0,
        'clean_test': 'ok', 'code': '2013PASS', 'decade code': 1.0,
        'domgross': 18.01, 'domgross_2013$': 18.01, 'imdb': 'tt1814621',
        'intgross': 18.01, 'intgross_2013$': 18.01,
        'period code': 1.0, 'test': 'ok', 'title': 'Admission', 'year': 2013},
        {'binary': 'FAIL', 'budget': 130.0, 'budget_2013$': 130.0,
        'clean_test': 'notalk', 'code': '2013FAIL', 'decade code': 1.0,
        'domgross': 60.52, 'domgross_2013$': 60.52, 'imdb': 'tt1815862',
        'intgross': 244.37, 'intgross_2013$': 244.37, 'period code': 1.0,
        'test': 'notalk', 'title': 'After Earth', 'year': 2013}]
        self.assertEqual(scaled_down, scaled_nine_and_ten)

    def test_revenues_per_budgets_trace(self):
        parsed_movies = self.parsed_movies
        self.assertEqual(revenues_per_budgets_trace['x'][0:10], [13.0, 45.66, 20.0, 61.0, 40.0, 225.0, 92.0, 12.0, 13.0, 130.0])
        self.assertEqual(revenues_per_budgets_trace['y'][0:10], [25.68, 13.61, 53.11, 75.61, 95.02, 38.36, 67.35, 15.32, 18.01, 60.52])
        self.assertEqual(revenues_per_budgets_trace['text'][0:10], ['21 &amp; Over',  'Dredd 3D', '12 Years a Slave', '2 Guns', '42', '47 Ronin',  'A Good Day to Die Hard', 'About Time',  'Admission',  'After Earth'])

    def test_highest_domestic_gross(self):
        self.assertEqual(highest_domestic_gross(movies)['title'], 'Star Wars')

    def test_outside_consultant_predicted_revenue(self):
        budget = 10
        self.assertEqual(outside_consultant_predicted_revenue(budget), 25)

    def test_movies_trace(self):
        self.assertEqual(consultant_estimated_revenues_trace['x'][0:10],[13.0, 45.66, 20.0, 61.0, 40.0, 225.0, 92.0, 12.0, 13.0, 130.0])
        self.assertEqual(consultant_estimated_revenues_trace['y'][0:10], [29.5, 78.49, 40.0, 101.5, 70.0, 347.5, 148.0, 28.0, 29.5, 205.0])
        self.assertEqual(consultant_estimated_revenues_trace['mode'], 'line')
        self.assertEqual(consultant_estimated_revenues_trace['name'], 'external consultant estimate')

    def test_error_for_consultant_model(self):
        american_hustle = {'binary': 'PASS', 'budget': 40.0, 'budget_2013$': 40.0, 'clean_test': 'ok',
         'code': '2013PASS', 'decade code': 1.0,
         'domgross': 148.43, 'domgross_2013$': 148.43, 'imdb': 'tt1800241', 'intgross': 249.48,
         'intgross_2013$': 249.48, 'period code': 1.0,
         'test': 'ok-disagree', 'title': 'American Hustle', 'year': 2013}
        self.assertEqual(error_for_consultant_model(american_hustle), 78.43)

    def test_rss_consultant(self):
        self.assertEqual(rss_consultant(scaled_movies), 23234357.68)

    def test_revenue_with_budget(self):
        self.assertEqual(revenue_with_year(25, 1997), 75.5)
        self.assertEqual(revenue_with_year(40, 1983), 71.0)

    def test_internal_consultant_estimated_revenue(self):
        self.assertEqual(internal_consultant_estimated_trace['x'][0:10],[13.0, 45.66, 20.0, 61.0, 40.0, 225.0, 92.0, 12.0, 13.0, 130.0])
        self.assertEqual(internal_consultant_estimated_trace['y'][0:10], [86.3, 120.726, 94.0, 139.10000000000002, 116.0, 319.5, 173.2, 85.2, 86.3, 215.0])
        self.assertEqual(internal_consultant_estimated_trace['mode'], 'markers')
        self.assertEqual(internal_consultant_estimated_trace['name'], 'internal consultant estimate')

    def test_rss_revenue_with_year(self):
        self.assertEqual(rss_revenue_with_year(scaled_movies), 25364329.23)

    def test_expected_revenue_per_budget(self):
        american_hustle = {'binary': 'PASS', 'budget': 40.0, 'budget_2013$': 40.0, 'clean_test': 'ok',
         'code': '2013PASS', 'decade code': 1.0, 'domgross': 148.43, 'domgross_2013$': 148.43, 'imdb': 'tt1800241',
         'intgross': 249.48, 'intgross_2013$': 249.48, 'period code': 1.0, 'test': 'ok-disagree',
         'title': 'American Hustle', 'year': 2013}
        self.assertEqual(expected_revenue_per_budget(american_hustle['budget_2013$']), 72.1)

    def test_regression_revenue_error(self):
        initial_regression_line = {'b': 0.5, 'm': 1.79}
        american_hustle = {'binary': 'PASS', 'budget': 40.0, 'budget_2013$': 40.0, 'clean_test': 'ok',
         'code': '2013PASS', 'decade code': 1.0, 'domgross': 148.43, 'domgross_2013$': 148.43, 'imdb': 'tt1800241',
         'intgross': 249.48, 'intgross_2013$': 249.48, 'period code': 1.0, 'test': 'ok-disagree',
         'title': 'American Hustle', 'year': 2013}
        self.assertEqual(regression_revenue_error(initial_regression_line['m'], initial_regression_line['b'], american_hustle), 76.33)

    def test_rss_trace(self):
        trace_values = {'mode': 'line', 'name': 'data',
        'text': [], 'x': [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
        'y': [23360190.02, 22710401.61, 22279030.46, 22066076.55,
        22071539.9, 22295420.5, 22737718.35, 23398433.45, 24277565.8,
        25375115.41]}
        self.assertEqual(trace_values, rss_trace)

    def test_b_gradient(self):
        self.assertEqual(b_gradient(1.79, 0.50, scaled_movies), 5.37)

    def test_m_gradient(self):
        self.assertEqual(m_gradient(1.79, 0.50, scaled_movies), 2520.59)

    def test_step_gradient(self):
        initial_regression_line = {'b': 0.5, 'm': 1.79}
        self.assertEqual(step_gradient(initial_regression_line['b'], initial_regression_line['m'], scaled_movies, .0001), {'b': 0.5, 'm': 1.54})

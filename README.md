
# Applying Gradient Descent Lab

### Introduction

In this lesson, let's put our knowledge about data science to the test. We will have access to functions in the [error](https://github.com/learn-co-curriculum/applying-gradient-descent-lab/blob/master/error.py), [graph](https://github.com/learn-co-curriculum/applying-gradient-descent-lab/blob/master/graph.py) and [linear equations](https://github.com/learn-co-curriculum/applying-gradient-descent-lab/blob/master/linear_equations.py) libraries that we previously wrote.

This is our task. We are an employee for *Good Lion Studios*. For *Good Lion*, our job is first to gather, explore, and format our data so that we can build a regression line of this data. Then we will work through various attempts of building out these regression lines. By the end of this lab, we should have a working version that we can proudly show to our manager.

### Learning Objectives

* Review how to use built-in functions, like filter and map, to clean data
* Evaluate the quality of regression lines using Residual Sum of Squares (RSS)
* Review how RSS changes with varying values of slope and y-intercept of a regression line
* Implement gradient descent to find a "best fit" regression line

This lesson is an opportunity to review the concepts explained in our introduction to machine learning section and practice what we recently learned about gradient descent to find an optimal regression line.

> ****Use the round method****: For many of the methods, we round down the return value to two decimal places. We can do so by using the **round** function, as in **round(12.1212, 2)** to round 12.1212 to 12.12. We did this to allow for slight differences between our results and expectations. So when we see our data differing from the expectation in the tests, check if using the **round** function helps.

### Determining Quality

#### Retrieve the data

First, let's get some movies from the FiveThirtyEight dataset [provided here](https://raw.githubusercontent.com/fivethirtyeight/data/master/bechdel/movies.csv).  The code below parses this data from the csv file and saves it to the `movies` variable as a list of dictionaries.


```python
import pandas

def parse_file(fileName):
    movies_df = pandas.read_csv(fileName)
    return movies_df.to_dict('records')

movies = parse_file('https://raw.githubusercontent.com/fivethirtyeight/data/master/bechdel/movies.csv')
```


```python
type(movies) # list
len(movies) # 1794
```

As you can see, this list is full of 1794 dictionaries, each one representing a movie.

#### Explore the data


Let's take a look at that first movie in the dataset.


```python
movies[0]
```

Here we can see the data available for each movie. The information most relevant for our task is:

1. `budget_2013$` is the budget adjusted for inflation in 2013 dollars
2. `domgross_2013$` is the domestic revenue adjusted for inflation in 2013 dollars
3. `intgross_2013$` is the international revenue adjusted for inflation in 2013 dollars

### Cleaning our data

#### 1. Handle missing data

Now, let's look at the values associated with these attributes.  The first movie looks good since it has nice data for each of these attributes. Unfortunately, the data for some other movies might not be so fun to play with. Let's remove the movies whose `domgross_2013` points to values of `nan`, which stands for "not a number". This data is missing.  There are only a few pieces of missing data here, so we can safely remove these movies without causing too much damage.


```python
import math
list(filter(lambda movie: math.isnan(movie['domgross_2013$']), movies))
```

Write a function called `remove_movies_missing_data` that returns the subset of movies that don't have `nan`.

> To do so, you can import the math library and make use of the `math.isnan` method.  More information about this method can be [found here](https://stackoverflow.com/questions/944700/how-can-i-check-for-nan-in-python).


```python
import math

def remove_movies_missing_data(movies):
    pass
```


```python
parsed_movies = remove_movies_missing_data(movies) or []
```

After writing the `remove_movies_missing_data` function, notice that we have reduced the number of movies down from 1794 to 1776 movies.


```python
len(parsed_movies) # 1776
```

Also, we can see that no movies with a `domgross_2013` value of `nan` are included.


```python
list(filter(lambda movie: math.isnan(movie['domgross_2013$']),parsed_movies)) # []
```

#### 2. Changing the scale of our data

Currently, our data has some very large numbers:


```python
movies[0]['budget']
```

It takes some time to figure out the number above is 13 million.  It would be frustrating to count all of the zeros whenever we come across another set of movie budgets and revenues.

To make things simpler, let's divide both our budget and revenue numbers for each movie by 1 million. It will make some of our future calculations easier to interpret. The attributes that we can scale down are `budget`, `budget_2013$`, `domgross`, `domgross_2013$`, `intgross`, and `intgross_2013$`.

Write a function called `scale_down_movie` that can take an element from our movies list and return that same movie but with the `budget`, `budget_2013$`, `domgross`, `domgross_2013$`, `intgross`, and `intgross_2013$` numbers all divided by 1 million and rounded to two decimal places.


```python
def scale_down_movie(movie):
    pass
```


```python
movies[0]
```


```python
scale_down_movie(movies[0])
```

Ok, now that we have a function to scale down our movies, lets `map` through all of our `parsed_movies` to return a list of `scaled_movies`. 


```python
def scale_down_movies(movies):
    pass
```


```python
first_ten_movies = parsed_movies[0:10]
first_ten_scaled = scale_down_movies(first_ten_movies) or []
first_ten_scaled[-2:]
#[{'year': 2013,
#  'imdb': 'tt1814621',
#  'title': 'Admission',
#  'test': 'ok',
#  'clean_test': 'ok',
#  'binary': 'PASS',
#  'budget': 13.0,
#  'domgross': 18.01,
#  'intgross': 18.01,
#  'code': '2013PASS',
#  'budget_2013$': 13.0,
#  'domgross_2013$': 18.01,
#  'intgross_2013$': 18.01,
#  'period code': 1.0,
#  'decade code': 1.0},
# {'year': 2013,
#  'imdb': 'tt1815862',
#  'title': 'After Earth',
#  'test': 'notalk',
#  'clean_test': 'notalk',
#  'binary': 'FAIL',
#  'budget': 130.0,
#  'domgross': 60.52,
#  'intgross': 244.37,
#  'code': '2013FAIL',
#  'budget_2013$': 130.0,
#  'domgross_2013$': 60.52,
#  'intgross_2013$': 244.37,
#  'period code': 1.0,
#  'decade code': 1.0}]
```


```python
scaled_movies = scale_down_movies(parsed_movies) or []
```

#### 3. Continue exploring the data

Now let's plot our dataset using Plotly to see how much money a movie made domestically in 2013 dollars given a budget in 2013 dollars. Create a trace called `revenues_per_budgets_trace` that plots this data.

> To do so, set `budget_2013$` as the $x$ values, and the `domgross_2013$` as the $y$ values. Set the text of the trace equal to a list of the movie titles, so that we can see which movie is associated with each point. All of the data should be coming from our `scaled_movies` variable.


```python
budgets = list(map(lambda movie: movie['budget_2013$'], scaled_movies))
domestic_revenues = list(map(lambda movie: movie['domgross_2013$'], scaled_movies))
titles = list(map(lambda movie: movie['title'], scaled_movies))
```

We'll check the first ten values of the `budgets`, `domestic_revenues`, and `titles` lists, but your trace should have an element for each of the `scaled_movies` in the dataset.


```python
budgets[0:10] # [13.0, 45.66, 20.0, 61.0, 40.0, 225.0, 92.0, 12.0, 13.0, 130.0]
```


```python
domestic_revenues[0:10] # [25.68, 13.61, 53.11, 75.61, 95.02, 38.36, 67.35, 15.32, 18.01, 60.52]
```


```python
titles[0:10] 
# ['21 &amp; Over',  'Dredd 3D', '12 Years a Slave', '2 Guns', '42', '47 Ronin',  'A Good Day to Die Hard',
# 'About Time',  'Admission',  'After Earth']
```

Once we have lists of these values, we are ready to create a trace. The following code creates a trace with the x values set as the `budgets`, the y values set as the `domestic_revenues`, and the text set as each of the movie `titles`.


```python
from graph import trace_values
revenues_per_budgets_trace = trace_values(budgets, domestic_revenues, text = titles)
```

> Once we have written the above code, we'll be ready to plot this data. Press shift + enter on the code below and you should see all movies in a graph.


```python
from graph import plot
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=True)

plot([revenues_per_budgets_trace])
```

Look at that one datapoint that earned well over 1.5 billion dollars.  What movie is that?

Write a function called `highest_domestic_gross` that finds the highest grossing movie given a list of movies.


```python
def highest_domestic_gross(movies):
    pass
```


```python
max_movie = highest_domestic_gross(scaled_movies) or {'title': 'some non movie'}
max_movie['title'] # 'Star Wars'
```

Huh, well we should've known.  Now let's zoom in on our dataset so that our plot no longer expands for just a few of the outliers. We will set the x-axis of our plot to go from zero to 300 million dollars, and the y-axis of our plot to go from zero to one billion dollars.


```python
from graph import build_layout
revenues_per_budgets_trace = trace_values(budgets, domestic_revenues, text = titles)
revenues_layout = build_layout(x_range = [0, 300], y_range = [0, 1000])
plot([revenues_per_budgets_trace], revenues_layout)
```

Ok, well at least we have a closer look at our data.  We still see Titanic up in the top right corner.

### Building our models

Ok, now that we have collected and explored this data, our company hired an outside consultant to create a model that predicts revenue for us. The consultant provided us with the following:

$$R(x) = 1.5*budget + 10$$

* where $x$ is a movie's budget in 2013 dollars, and $R(x)$ is the expected revenue in 2013 dollars.

Write a function called `outside_consultant_predicted_revenue` that, provided a budget, returns the expected revenue according to the outside consultant's formula.


```python
def outside_consultant_predicted_revenue(budget):
    pass
```

Let's plot the consultant estimated revenue to see visually if his estimates line up.  We will call this trace `external consultant estimate`.


```python
budgets = list(map(lambda movie: movie['budget_2013$'], scaled_movies))
domestic_revenues = list(map(lambda movie: movie['domgross_2013$'], scaled_movies))
titles = list(map(lambda movie: movie['title'], scaled_movies))

consultant_estimated_revenues = list(map(lambda budget: outside_consultant_predicted_revenue(budget),budgets))
consultant_estimated_revenues_trace = trace_values(budgets, consultant_estimated_revenues, mode='lines', name = 'external consultant estimate')
```


```python
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=True)

from graph import trace_values, m_b_trace, plot

plot([revenues_per_budgets_trace, consultant_estimated_revenues_trace], revenues_layout)
```

Overall, the model doesn't look too bad.  However, we can calculate the RSS to quantify how accurate his model really is.

Let's write a method called `error_for_consultant_model` which takes in a budget of a movie in our dataset, and returns the difference between the movie's gross domestic revenue in 2013 dollars, and the prediction from the consultant's model.


```python
def error_for_consultant_model(movie):
    pass
```


```python
american_hustle = {'binary': 'PASS', 'budget': 40.0, 'budget_2013$': 40.0, 'clean_test': 'ok',
         'code': '2013PASS', 'decade code': 1.0, 'domgross': 148.43, 'domgross_2013$': 148.43, 'imdb': 'tt1800241',
         'intgross': 249.48, 'intgross_2013$': 249.48, 'period code': 1.0, 'test': 'ok-disagree',
         'title': 'American Hustle', 'year': 2013}
error_for_consultant_model(american_hustle) # 78.43
```

Once haven written a formula that calculates the error for the consultant's model provided a budget, we can write a method that calculates the RSS for the consultant's model.  When we move onto compare our consultant's model with others, we'll then have a metric for comparison.


```python
def rss_consultant(movies):
    pass
```


```python
rss_consultant(scaled_movies) # 23234357.68
```

Ok, we'll find out if this number is any good later, but for right now let's just say that our RSS is good enough.  Use the derivative to write a function that provided a budget, returns the $\frac{\Delta R}{\Delta x}$ according to the consultant's model.  Remember that our consultant's model is $R(x) = 1.5x + 10$ where $x$ is a budget, and $R(x)$ is an expected revenue.

### A new model

Now imagine a data scientist in your company wants to take a crack at his own model for predicting a movie's revenue.  The data scientist notices, that in general, movies tend to make more money per year.


```python
from graph import build_layout
years = list(map(lambda movie: movie['year'],movies))
years_and_revenues = trace_values(years, domestic_revenues, text = titles)
years_layout = build_layout(y_range = [0, 550])
plot([years_and_revenues], years_layout)
```

So the data scientist comes up with a new model, to indicate a movie's expected revenue is 1.5 million for every year after 1965 plus 1.1 times the movie's budget.  Write a function called `revenue_with_year` that takes as arguments `budget` and `year` and returns expected revenue.  


```python
def revenue_with_year(budget, year):
    pass
```


```python
revenue_with_year(25, 1997) # 75.5
revenue_with_year(40, 1983) # 71.0
```

Notice that this model has **two variables**, the budget and year, and therefore is not a line function.  Let's compare these models by plotting the actual revenues and budgets, the prior `external consultant estimate` line trace, and the `internal consultant estimate` based upon this model's estimates.  Since this model doesn't produce a line, we will set the mode for `internal_consultant_estimated_trace` to 'markers'.


```python
budgets = list(map(lambda movie: movie['budget_2013$'], scaled_movies))
domestic_revenues = list(map(lambda movie: movie['domgross_2013$'], scaled_movies))
titles = list(map(lambda movie: movie['title'], scaled_movies))

internal_consultant_estimated_revenues = list(map(lambda movie: revenue_with_year(movie['budget_2013$'], movie['year']),scaled_movies))
internal_consultant_estimated_trace = trace_values(budgets, internal_consultant_estimated_revenues, mode='markers', name = 'internal consultant estimate')
```


```python
internal_consultant_estimated_trace['x'][0:10] # [13.0, 45.66, 20.0, 61.0, 40.0, 225.0, 92.0, 12.0, 13.0, 130.0]
internal_consultant_estimated_trace['y'][0:10] # [86.3, 120.726, 94.0, 139.10000000000002, 116.0, 319.5, 173.2, 85.2, 86.3, 215.0]
internal_consultant_estimated_trace['mode'] # 'markers'
```


```python
from plotly.offline import iplot, init_notebook_mode
init_notebook_mode(connected=True)

from graph import trace_values, m_b_trace, plot
plot([revenues_per_budgets_trace, consultant_estimated_revenues_trace, internal_consultant_estimated_trace], revenues_layout)
```

Although the `external consultant model` isn't a line, it still seems to match our data fairly well.  Let's find out how well.  Even though it is not a line, we can still calculate the RSS for this model.  Write a function called `rss_revenue_with_year` that returns the Residual Sum of Squares associated with the `revenue_with_year` model for the `scaled_movies` dataset.  The `squared_error_revenue_with_year` function can be used to return the squared error of the model associated with just a single movie.


```python
def squared_error_revenue_with_year(movie):
    pass

def rss_revenue_with_year(movies):
    pass
```


```python
rss_revenue_with_year(scaled_movies) # 25364329.23
```

The RSS here is $25,364,329.23$ as opposed to the RSS of $23,234,357.68$ from the external consultant's model.  According to RSS, this model isn't as accurate as the previous model. Still, it isn't bad enough to ignore completely.

### Our initial regression line, and improving upon it

Now that we have evaluated the models of an outside consultant and an internal consultant, it's time to see if we can do any better.  Let's go.

We have our dataset. Let's begin with an initial regression line that sets $b = .5$ and $m = 1.79$.


```python
from linear_equations import build_regression_line
budgets = list(map(lambda movie: movie['budget_2013$'], scaled_movies)) or [1, 2]
domestic_revenues = list(map(lambda movie: movie['domgross_2013$'], scaled_movies)) or [1, 2]
```


```python
initial_regression_line = {'b': 0.5, 'm': 1.79}
```

Using values for $m$ and $b$ from our initial regression line, we can write `expected_revenue_per_budget` that returns the expected revenue provided a budget.


```python
def expected_revenue_per_budget(budget):
    pass
```


```python
budget = american_hustle['budget_2013$'] # 40.0
expected_revenue_per_budget(budget) # 72.1
```

Now this initial regression line was not very sophisticated. We simply drew a line between the points with the lowest and highest $x$ values.

Let's plot our initial regression line along our dataset to get a sense of the accuracy of this first line.


```python
budgets = list(map(lambda movie: movie['budget_2013$'], scaled_movies))
estimated_revenues = list(map(lambda budget: expected_revenue_per_budget(budget), budgets))
len(estimated_revenues)
initial_regression_trace = trace_values(budgets, estimated_revenues, mode = 'lines', name = 'initial regression trace')
```


```python
plot([revenues_per_budgets_trace, initial_regression_trace], revenues_layout)
```

By now we should be able to guess the next step: quantify how well this line matches our data.  We'll write a function called `regression_revenue_error` that, provided a movie and an `m` and `b` value of a regression line, returns the difference between our `initial_regression_lines`'s expected revenue and the actual revenue error.


```python
def regression_revenue_error(m, b, movie):
    pass
```


```python
initial_regression_line
```


```python
american_hustle = {'binary': 'PASS', 'budget': 40.0, 'budget_2013$': 40.0, 'clean_test': 'ok',
         'code': '2013PASS', 'decade code': 1.0, 'domgross': 148.43, 'domgross_2013$': 148.43, 'imdb': 'tt1800241',
         'intgross': 249.48, 'intgross_2013$': 249.48, 'period code': 1.0, 'test': 'ok-disagree',
         'title': 'American Hustle', 'year': 2013}

regression_revenue_error(initial_regression_line['m'], initial_regression_line['b'], american_hustle)
# 76.33
```

Ok, now plot the cost curve from changing values of $m$ from $1.0$ to $1.9$.

> We don't ask you to write a function for calculating the RSS, as you already wrote one in the error library which is available to you, and you can see used here.


```python
from error import residual_sum_squares
residual_sum_squares(budgets, domestic_revenues, initial_regression_line['m'], initial_regression_line['b'])
```

But we do ask you to plot a cost curve from 1.0, to 1.9 using that `residual_sum_squares` function. We start off with a list of values of m from 1.0 to 1.9, assigned to `m_range` below.


```python
large_m_range = list(range(10, 20))
m_range = list(map(lambda m_value: m_value/10,large_m_range))
```

Now we need to calculate a list of RSS values associated with each value in the `m_range`.


```python
cost_values = list(map(lambda m_value: round(residual_sum_squares(budgets, domestic_revenues, m_value, initial_regression_line['b']), 2),m_range))
```


```python
from graph import trace_values
rss_trace = trace_values(x_values=m_range, y_values=cost_values, mode = 'lines')
```


```python
rss_trace
```


```python
plot([rss_trace])
```

Ok, so based on this, it appears that with our $b = 0.5$, the slope of our regression line that produces the lowest error is between $1.3$ and $1.4$. In fact if we replace our initial line value of $m$ with $1.3$, we see that our RSS does in fact decline from our previous value of $24,179,824$.


```python
residual_sum_squares(budgets, domestic_revenues, 1.3, initial_regression_line['b'])
```

### Changing multiple variables

Ok, now it's time to move beyond testing the accuracy of the line with changing only a single variable.  We need to play with both variables to find the 'best fit regression line'. As we know, the technique for that is to use gradient descent.

Remember that we derived our gradient formulas by starting with our cost function, and saying the RSS is a function of our $m$ and $b$ variables:

$$J(m,b) = \sum_{i = 1}^n(y_i - (mx_i + b))^2 $$

From the above formula for our cost curve, we found the gradient descent of the cost function, as that is used to find the incremental changes to decrease RSS. We do this mathematically, by taking the partial derivative with respect to $m$ and $b$.

$$ \frac{dJ}{db}J(m,b) = -2\sum_{i = 1}^n(y_i - (mx_i + b)) = -2\sum_{i = 1}^n \epsilon_i $$
$$ \frac{dJ}{dm}J(m,b) = -2\sum_{i = 1}^n x(y_i - (mx_i + b)) = -2\sum_{i = 1}^n x_i*\epsilon_i$$



Looking at our top function $\frac{dJ}{dm}$, we see that it equals negative 2, multiplied by the sum of the errors for a provided $m$ and $b$ values relative to our dataset.  And luckily for us, we already have a function called `regression_revenue_error` that returns the error at a given point when provided our $m$ and $b$ values.

Our task now is two write a function called `b_gradient` that takes in values of $m$, $b$ and our (scaled) movies, and returns the `b` gradient, which is -2 times the sum of the errors for the dataset.


```python
def b_gradient(m, b, movies):
    pass
```


```python
b_gradient(1.79, 0.50, scaled_movies) # 5.37
```

Next, write a function called `m_gradient` that returns the `m` gradient for values of $m$, $b$, and a list of movies.


```python
def m_gradient(m, b, movies):
    pass
```


```python
m_gradient(1.79, 0.50, scaled_movies) # 2520.59
```

> Notice that the `m_gradient` is significantly larger than the `b_gradient`. This makes sense since the `m_gradient` formula is similar to the `b_gradient` formula, except that its output is also multiplied by the corresponding $x$ value.

Ok, now we just wrote two functions that tell us how to update the corresponding values of $m$ and $b$.  Our next step is to write a function called `step_gradient` that will use these functions to take the step down along our cost curve.

Remember that with each step we want to move our `current_b` value in the negative direction of calculated `b_gradient`, and want to move our `current_m` value in the negative direction of the calculated `m_gradient`.  

`current_m` = `old_m` $ -  \eta(-2*\sum_{i=1}^n x_i*\epsilon_i )$

`current_b` =  `old_b` $ - \eta( -2*\sum_{i=1}^n \epsilon_i )$

The `step_gradient` function would take as arguments the `b_current`, `m_current`, the list of scaled movies, and a learning rate, and returns a newly calculated `b_current` and `m_current` with a dictionary of keys `b` and `m` that point to the current values.   


```python
def step_gradient(b_current, m_current, movies, learning_rate):
    pass
```


```python
initial_regression_line # {'b': 0.5, 'm': 1.79}
```

Then let's see how our formula changes over time using gradient descent.


```python
step_gradient(initial_regression_line['b'], initial_regression_line['m'], scaled_movies, .0001)
```

Now write a function that can operate given a set of 100 iterations and start from our `initial_regression_line`.


```python
# set our initial step with m and b values, and the corresponding error.
def generate_steps(m, b, number_of_steps, movies, learning_rate):
    pass

#     iterations = []
#     for i in range(number_of_steps):
#         iteration = step_gradient(b, m, movies, learning_rate)
#         # {'b': value, 'm': value}
#         b = iteration['b']
#         m = iteration['m']
#         # update values of b and m
#         iterations.append(iteration)
#     return iterations
```


```python
iterations = generate_steps(initial_regression_line['b'], initial_regression_line['m'], 100, scaled_movies, .0001) or [{'m': 'uncomment generate_steps method', 'b': 'uncomment generate_steps method '}]
```

And we can see how this changes over time.


```python
def to_line(m, b):
    initial_x = 0
    ending_x = 500
    initial_y = m*initial_x + b
    ending_y = m*ending_x + b
    return {'data': [{'x': [initial_x, ending_x], 'y': [initial_y, ending_y]}]}

frames = list(map(lambda iteration: to_line(iteration['m'], iteration['b']),iterations))
```


```python
from plotly.offline import init_notebook_mode, iplot
from IPython.display import display, HTML

init_notebook_mode(connected=True)

budgets = list(map(lambda movie: movie['budget_2013$'], scaled_movies))
domestic_revenues = list(map(lambda movie: movie['domgross_2013$'], scaled_movies))

figure = {'data': [{'x': [0], 'y': [0]}, {'x': budgets, 'y': domestic_revenues, 'mode': 'markers'}],
          'layout': {'title': 'Regression Line',
                     'updatemenus': [{'type': 'buttons',
                                      'buttons': [{'label': 'Play',
                                                   'method': 'animate',
                                                   'args': [None]}]}]
                    },
          'frames': frames}
iplot(figure)
```

Finally, let's calculate the RSS associated with our formula as opposed to the other.


```python
iterations[-1] # {'b': 1.8, 'm': 1.37}
```


```python
residual_sum_squares(budgets, domestic_revenues, iterations[-1]['m'], iterations[-1]['b'])
```

Using this last iteration, we have an RSS $21,982,786$, better than all previous models - and we have the data, and knowledge to prove it:


```python
external_consultant_model = rss_consultant(scaled_movies)
internal_consultant_model = rss_revenue_with_year(scaled_movies)
our_regression_model = residual_sum_squares(budgets, domestic_revenues, iterations[-1]['m'], iterations[-1]['b'])
```


```python
external_consultant_model # 23234357.68
internal_consultant_model # 25364329.23
our_regression_model      # 21982786.34
```

Nice work!

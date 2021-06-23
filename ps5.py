# -*- coding: utf-8 -*-
# Problem Set 5: Experimental Analysis
# Name: 
# Collaborators (discussion):
# Time:

import pylab
import re

# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHILADELPHIA',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'SAN JUAN',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

TRAINING_INTERVAL = range(1961, 2010)
TESTING_INTERVAL = range(2010, 2016)

"""
Begin helper code
"""
class Climate(object):
    """
    The collection of temperature records loaded from given csv file
    """
    def __init__(self, filename):
        """
        Initialize a Climate instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature
            
        f.close()

    def get_yearly_temp(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a 1-d pylab array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return pylab.array(temperatures)

    def get_daily_temp(self, city, month, day, year):
        """
        Get the daily temperature for the given city and time (year + date).

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified time (year +
            date) and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]


def se_over_slope(x, y, estimated, model):
    """
    For a linear regression model, calculate the ratio of the standard error of
    this fitted curve's slope to the slope. The larger the absolute value of
    this ratio is, the more likely we have the upward/downward trend in this
    fitted curve by chance.
    
    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by a linear
            regression model
        model: a pylab array storing the coefficients of a linear regression
            model

    Returns:
        a float for the ratio of standard error of slope to slope
    """
    assert len(y) == len(estimated)
    assert len(x) == len(estimated)
    EE = ((estimated - y)**2).sum()
    var_x = ((x - x.mean())**2).sum()
    SE = pylab.sqrt(EE/(len(x)-2)/var_x)
    return round(SE/model[0], 5)

"""
End helper code
"""


def generate_models(x, y, degs):
    """
    Generate regression models by fitting a polynomial for each degree in degs
    to points (x, y).

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        degs: a list of degrees of the fitting polynomial

    Returns:
        a list of pylab arrays, where each array is a 1-d array of coefficients
        that minimizes the squared error of the fitting polynomial
    """
    list_arrays = []
    for degree in degs:
        coefficients = pylab.array(pylab.polyfit(x, y, degree))
        list_arrays.append(coefficients)

    return list_arrays
# return [pylab.array(pylab.polyfit(x, y, deg)) for deg in degs]


def r_squared(y, estimated):
    """
    Calculate the R-squared error term.
    
    Args:
        y: 1-d pylab array with length N, representing the y-coordinates of the
            N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the R-squared error term
    """

    assert len(y) == len(estimated)

    diff = []
    for i in range(len(y)):
        diff.append((y[i] - estimated[i])**2)

    mean = pylab.mean(y)
    means = []
    for i in range(len(y)):
        means.append((y[i] - mean)**2)

    return round(1 - sum(diff) / sum(means), 5)


# TODO
def evaluate_models_on_training(x, y, models):
    """
    For each regression model, compute the R-squared value for this model with the
    standard error over slope of a linear regression line (only if the model is
    linear), and plot the data along with the best fit curve.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        R-square of your model evaluated on the given data points,
        and SE/slope (if degree of this model is 1 -- see se_over_slope). 

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    best_r = None
    best_est = None
    best_mod = None

    for model in models:
        estimates = pylab.polyval(model, x)
        r = r_squared(y, estimates)
        if best_r is None or best_r < r:
            best_r = r
            best_est = estimates
            best_mod = model

    SE_Slope = "--" if len(best_mod) == 1 else str(se_over_slope(x, y, best_est, best_mod))

    pylab.xlabel("year")
    pylab.ylabel("temperature")
    pylab.plot(x, y, 'bo', label='data')
    pylab.plot(x, best_est, '-r', label='est')
    pylab.legend(loc="upper right")
    pylab.title(str(len(best_mod) - 1) + " " + str(best_r) + " " + SE_Slope)
    pylab.show()


def gen_cities_avg(climate, multi_cities, years):
    """
    Compute the average annual temperature over multiple cities.

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to average over (list of str)
        years: the range of years of the yearly averaged temperature (list of
            int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the average annual temperature over the given
        cities for a given year.
    """
    averages = []
    for year in years:
        sum_temp = 0
        length = 0
        for city in multi_cities:
            temps = climate.get_yearly_temp(city, year)
            sum_temp += sum(temps)
            length += len(temps)

        averages.append(sum_temp / length)

    p = pylab.array(averages)
    return p


def moving_average(y, window_length):
    """
    Compute the moving average of y with specified window length.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        window_length: an integer indicating the window length for computing
            moving average

    Returns:
        an 1-d pylab array with the same length as y storing moving average of
        y-coordinates of the N sample points
    """
    averages = []
    for i in range(len(y)):
        start = max(0, i - window_length + 1)
        avg = sum(y[start: i + 1]) / float(i + 1 - start)
        averages.append(avg)
    return pylab.array(averages)


def rmse(y, estimated):
    """
    Calculate the root mean square error term.

    Args:
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d pylab array of values estimated by the regression
            model

    Returns:
        a float for the root mean square error term
    """
    assert len(y) == len(estimated)
    diff = 0
    n = len(y)
    for i in range(n):
        diff += (y[i] - estimated[i]) ** 2

    return (diff / n) ** 0.5


def gen_std_devs(climate, multi_cities, years):
    """
    For each year in years, compute the standard deviation over the averaged yearly
    temperatures for each city in multi_cities. 

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to use in our std dev calculation (list of str)
        years: the range of years to calculate standard deviation for (list of int)

    Returns:
        a pylab 1-d array of floats with length = len(years). Each element in
        this array corresponds to the standard deviation of the average annual 
        city temperatures for the given cities in a given year.
    """
    std_years = []
    for year in years:
        all_cities_yearly_temp = []
        for city in multi_cities:
            yearly_temp = climate.get_yearly_temp(city, year)
            all_cities_yearly_temp.append(yearly_temp)
        all_cities_yearly_temp = pylab.array(all_cities_yearly_temp)  # converting list to an array.
        daily_mean = all_cities_yearly_temp.mean(axis=0)  # calculating mean for each day from all the city arrays.
        std_dev = pylab.std(daily_mean)  # calculating standard deviation across the year.
        std_years.append(std_dev)
    return pylab.array(std_years)

def evaluate_models_on_testing(x, y, models):
    """
    For each regression model, compute the RMSE for this model and plot the
    test data along with the model’s estimation.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        RMSE of your model evaluated on the given data points. 

    Args:
        x: an 1-d pylab array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d pylab array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a pylab array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    best_rmse = None
    best_est = None

    for model in models:
        estimates = pylab.polyval(model, x)
        current_rmse = rmse(y, estimates)
        if best_rmse is None or current_rmse < best_rmse:
            best_rmse = current_rmse
            best_est = estimates


    pylab.xlabel("year")
    pylab.ylabel("temperature")
    pylab.plot(x, y, 'bo', label='data')
    pylab.plot(x, best_est, '-r', label='est')
    pylab.legend(loc="upper right")
    pylab.title(str(best_rmse))
    pylab.show()


if __name__ == '__main__':

    # Part A.4
    climate = Climate("data.csv")

    '''
    samples = []
    city = "NEW YORK"
    month = 1
    day = 10
    for year in TRAINING_INTERVAL:
        temperature = climate.get_daily_temp(city, month, day, year)
        samples.append(temperature)

    samples = pylab.array(samples)
    coef = [1]
    model = generate_models(pylab.array(TRAINING_INTERVAL), samples, coef)
    evaluate_models_on_training(pylab.array(TRAINING_INTERVAL), samples, model)
    '''
    '''
    samples = []
    city = "NEW YORK"
    for year in TRAINING_INTERVAL:
        temperatures = climate.get_yearly_temp(city, year)
        samples.append(sum(temperatures) / len(temperatures))
    samples = pylab.array(samples)
    coef = [1]
    model = generate_models(pylab.array(TRAINING_INTERVAL), samples, coef)
    evaluate_models_on_training(pylab.array(TRAINING_INTERVAL), samples, model)
    '''

    # Part B
    '''
    samples = gen_cities_avg(climate, CITIES, TRAINING_INTERVAL)
    coef = [1]
    model = generate_models(pylab.array(TRAINING_INTERVAL), samples, coef)
    evaluate_models_on_training(pylab.array(TRAINING_INTERVAL), samples, model)
    '''
    # Part C

    '''
    samples = gen_cities_avg(climate, CITIES, TRAINING_INTERVAL)
    samples = moving_average(samples, 5)
    coef = [1]
    model = generate_models(pylab.array(TRAINING_INTERVAL), samples, coef)
    evaluate_models_on_training(pylab.array(TRAINING_INTERVAL), samples, model)
    '''
    # Part D.2

    '''
    samples = gen_cities_avg(climate, CITIES, TRAINING_INTERVAL)
    samples = moving_average(samples, 5)
    coef = [1, 2, 20]
    model = generate_models(pylab.array(TRAINING_INTERVAL), samples, coef)
    evaluate_models_on_testing(pylab.array(TRAINING_INTERVAL), samples, model)

    temperatures = gen_cities_avg(climate, CITIES, TESTING_INTERVAL)
    temperatures = moving_average(temperatures, 5)
    evaluate_models_on_testing(pylab.array(TESTING_INTERVAL), temperatures, model)
    '''
    # Part E

    std_devs = gen_std_devs(climate, CITIES, TRAINING_INTERVAL)
    std_devs = moving_average(std_devs, 5)
    models = generate_models(TRAINING_INTERVAL, std_devs, [1])
    evaluate_models_on_training(pylab.array(TRAINING_INTERVAL), std_devs, models)

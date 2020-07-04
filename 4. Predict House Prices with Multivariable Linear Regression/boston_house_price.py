from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np

# gather data
boston_data = load_boston()
df = pd.DataFrame(data=boston_data.data, columns=boston_data.feature_names)

features = df.drop(['INDUS', 'AGE'], axis=1)
log_price = np.log(boston_data.target)
log_price = pd.DataFrame(log_price, columns=['PRICE'])

CRIM_INX = 0
ZN_INX = 1
CHAS_INX = 2
RM_INX = 4
PTRATIO_INX = 8

property_stats = features.mean().values.reshape(1, 11)

reg = LinearRegression().fit(features, log_price)
fitted_values = reg.predict(features)

rsquare = reg.score(features, log_price)
mse = mean_squared_error(log_price, reg.predict(features))
rmse = np.sqrt(mse)

def get_log_estimate(number_room,
                     student_per_classroom,
                     is_next_to_river=False,
                     high_confidence=True):
    # config property
    property_stats[0][RM_INX] = number_room
    property_stats[0][PTRATIO_INX] = student_per_classroom
    property_stats[0][CHAS_INX] = 1 if is_next_to_river else 0

    # make prediction
    log_estimate = reg.predict(property_stats)[0][0]

    # calculate the range
    if high_confidence:  # 2 std
        upper_bound = log_estimate + 2*rmse
        lowwer_cound = log_estimate - 2*rmse
    else:  # one standard
        upper_bound = log_estimate + 1*rmse
        lowwer_cound = log_estimate - 1*rmse

    return log_estimate, upper_bound, lowwer_cound

def get_dollar_estimate(number_room, student_per_classroom, is_next_to_river=False, high_confidence=True):
    """ Estimate price in boston 
    Keyword arguments:
    number_room -- number of room in the property
    student_per_classroom -- number of students per teacher in the classroom for the school area
    is_next_to_river -- True if the property is next to the river
    high_confidence -- confidence on value price
    """
    if number_room < 1 or student_per_classroom < 1:
        return
    ZILLOW_MEDIAN_PRICE = 583.3
    scale = ZILLOW_MEDIAN_PRICE/np.median(boston_data.target)
    log_estimate, upper_bound, lowwer_bound = get_log_estimate(number_room=number_room,
                                                               student_per_classroom=student_per_classroom,
                                                               is_next_to_river=is_next_to_river,
                                                               high_confidence=high_confidence)
    # convert to today's price
    dollar_estimate = np.e**log_estimate * 1000 * scale
    dollar_estimate = dollar_estimate.round(-3)  
    upper_bound = np.e**upper_bound * 1000 * scale
    upper_bound = upper_bound.round(-3)
    lowwer_bound = np.e**lowwer_bound * 1000 * scale
    lowwer_bound = lowwer_bound.round(-3)    

    print('dollar estimate : {} $\nupper bound : {} $\nlowwer bound : {} $'.format(dollar_estimate, upper_bound, lowwer_bound))


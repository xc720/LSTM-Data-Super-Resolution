# LSTM-Data-Super-Resolution (2nd-year Summer Vacation)
# Undergraduate Research Opportunity Program
The target of energy data super-resolution is to estimate a high-resolution energy timeseries data for a low-resolution energy time-series data. For example, given 5-minute
data, in which each 5-minute data is the average of the corresponding 1-minute data,
we will try to predict the 1-minute data by our data super-resolution model.
Just like image super resolution, we normally assume neighbouring pixels in an image
to have similar values. I think that there may exist some similarities in the training set
to values that we need to predict. Thus the original model is inspired by a time-seriesdata prediction model.
All the work carried out are based on the fact that the researched data are from only one
house and energy consumption on weekdays are similar, which will probably results
inaccuracy if our model is used in other houses.

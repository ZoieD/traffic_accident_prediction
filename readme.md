# Traffic Predict

## python version 
3.6

## map
````
// road map
https://data.vicroads.vic.gov.au/arcgis/rest/services/HeavyVehicles/OSOM_SCHEME_MAP_NETWORK_D6/FeatureServer

// signal
https://services2.arcgis.com/18ajPSI0b3ppsmMt/arcgis/rest/services/traffic_lights/FeatureServer

````

## score
- average_precision_score：AP that calculates the predicted value
- f1_score: Calculate the F1 value, also known as the balanced F-score or F-meature
- precision_recall_curve：Precision-recall pairs that calculate different probability thresholds
- precision_score： calculates precision
- recall_score： calculates recall
- roc_auc_score: 

## Traffic Accident Prediction

#### Data Collection
- Data from gov of Victoria
- get weather from http://api.weather.com to get history weather data

#### Data Processing
- Feature includes accidents location, weather condition, road infrastructure, car and driver information. when accident happen, y = 1
- Simulate the datasets for the situations that accidents not happen in all conditions, y = 0


#### Data Training
- Using Tensorflow to train, the result isn't good enough 
- Using XGboost, the result better.

#### On the whole, the result is not so good, because the training data set too small and the accuracy of the datasets need to be improved.

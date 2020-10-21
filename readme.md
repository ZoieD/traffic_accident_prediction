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

## 交通事故预测

#### 收集数据
- Data from gov of 维州
- get weather from http://api.weather.com to get history weather data

#### 整理数据
- 整理每件交通事故发生地，对应的天气情况，路面信息，驾驶员信息，车辆信息等进行训练 y为1
- 模拟出不同时间，不同地点，相同的条件，不发生事故的点，y为0

#### 训练数据
- 使用tensorflow进行training，得到的结果不是很好
- 使用xgboost进行training，得到的结果可以

#### 训练结果总体并没有达到目标，数据量太小，数据的准确性不够。
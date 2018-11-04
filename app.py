# contains the code for the front end webapp
from flask import Flask, render_template
from flask_bootstrap import Bootstrap
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

app = Flask(__name__)
Bootstrap(app)

#
#def mostLikelyDuration(lat, lon, time):
#   # This method uses the KNN model to predict how long someone would need a bike based on
#    # where they are and what time it is
#    colsOfInterest = ["Duration", "Start Time", "Starting Station Latitude", "Starting Station Longitude"]
#    trainingFrame = pd.read_csv('./data/metro-bike-share-trip-data.csv')
#    trainingFrame = pd.DataFrame(trainingFrame, columns=colsOfInterest)
#    timeInSeconds = []
#    for i in range(len(trainingFrame.index)):
#        format = "%Y-%m-%dT%H:%M:%S"
#        t = datetime.strptime(trainingFrame["Start Time"].at[i], format)
#        timeInSeconds.append((timedelta(hours= t.hour, minutes=t.minute, seconds=t.second).total_seconds()))
#    trainingFrame["Start Time"] = timeInSeconds
#
#    lat = float(lat)
#    lon = float(lon)
#    format = "%H:%M:%S"
#    t = datetime.strptime(time, format)
#    time = timedelta(hours = t.hour, minutes = t.minute, seconds=t.second).total_seconds()
#
#    input = [[lat,lon, time]]
#    xtrain = trainingFrame.iloc[:,:1].values
#    ytrain = trainingFrame.iloc[:,1:].values
#
#    scaler = StandardScaler()
#    scaler.fit(xtrain)
#    xtrain = scaler.transform(xtrain)
#    print(xtrain)
#    input = scaler.transform(input)
#    print(input)

#    classifier = KNeighborsClassifier(n_neighbors=7)
#    classifier.fit(xtrain, ytrain)


#    output = classifier.predict(input)
#    print(output)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    #mostLikelyDuration(0,0,"12:12:12")
    app.run(debug=True)


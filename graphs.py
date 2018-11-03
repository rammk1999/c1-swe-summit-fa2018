# creates all the visuals for the webapp

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime

def filter_time(time):
    datetime_format = "%Y-%m-%dT%H:%M:%S"
    return datetime.strptime(time, datetime_format)

def popular_start_station():
    dataFrame = pd.read_csv('./data/metro-bike-share-trip-data.csv', low_memory=False)
    startStations = dataFrame['Starting Station ID']
    startStations = startStations.astype(str)
    startStationsSet = sorted(set(startStations))
    startStationsSet = startStationsSet[:-1] ## removes nulls
    usesPerStation = np.zeros(len(startStationsSet))
    for i in range(len(usesPerStation)):
        for station in startStations:
            if startStationsSet[i] == station:
                usesPerStation[i] += 1
    topTenInds = usesPerStation.argsort()[-10:][::-1]
    topTenStarts = np.zeros(len(topTenInds))
    topTenUsages = np.zeros(len(topTenInds))

    for i in range(len(topTenInds)):
        topTenStarts[i] = startStationsSet[topTenInds[i]]
    for i in range(len(topTenInds)):
        topTenUsages[i] = usesPerStation[topTenInds[i]]

    plt.figure()
    sns.set_style("dark")
    sns.set_style("ticks")
    sns.barplot(x=topTenStarts, y=topTenUsages)
    plt.xlabel("Starting Station ID")
    plt.ylabel("Number of Uses")
    plt.tight_layout();
    plt.savefig("static/graphs/popularStarts.png", format="png")

def popular_end_station():
    dataFrame = pd.read_csv('./data/metro-bike-share-trip-data.csv', low_memory=False)
    endStations = dataFrame['Ending Station ID']
    endStations = endStations.astype(str)
    endStationsSet = sorted(set(endStations))
    endStationsSet = endStationsSet[:-1] # removes nulls
    usesPerStation = np.zeros(len(endStationsSet))
    for i in range(len(usesPerStation)):
        for station in endStations:
            if endStationsSet[i] == station:
                usesPerStation[i] += 1
    topTenInds = usesPerStation.argsort()[-10:][::-1]
    topTenEnds = np.zeros(len(topTenInds))
    topTenUsages = np.zeros(len(topTenInds))

    for i in range(len(topTenInds)):
        topTenEnds[i] = endStationsSet[topTenInds[i]]
    for i in range(len(topTenInds)):
        topTenUsages[i] = usesPerStation[topTenInds[i]]

    plt.figure()
    sns.set_style("dark")
    sns.set_style("ticks")
    sns.barplot(x=topTenEnds, y=topTenUsages)
    plt.xlabel("Ending Station ID")
    plt.ylabel("Number of Uses")
    plt.tight_layout();
    plt.savefig("static/graphs/popularEnds.png", format="png")

def popular_stations():
    popular_start_station()
    popular_end_station()

def bike_sharing_breakdown():
    dataFrame = pd.read_csv('./data/metro-bike-share-trip-data.csv', low_memory=False)
    totalRides = len(dataFrame.index)
    ridePassTypes = dataFrame['Passholder Type']
    ridePassTypes = ridePassTypes.astype(str)
    regularPassTypes = ridePassTypes[ridePassTypes != "Walk-up"]
    regularRiders = len(regularPassTypes.index)
    percentRegulars = regularRiders/totalRides



def generate_graphs():
    popular_stations()


if __name__ == '__main__':
    #generate_graphs()
    bike_sharing_breakdown()

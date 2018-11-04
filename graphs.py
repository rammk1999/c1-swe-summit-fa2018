# creates all the visuals for the webapp

from math import radians, cos, sin, asin, sqrt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from datetime import datetime

def filter_time(time):
    datetime_format = "%Y-%m-%dT%H:%M:%S"
    return datetime.strptime(time, datetime_format)

def popular_stations():
    dataFrame = pd.read_csv('./data/metro-bike-share-trip-data.csv', low_memory=False)
    startStations = dataFrame['Starting Station ID']
    startStations = startStations.astype(str)
    startStationsSet = sorted(set(startStations)) # get a single list of all the start stations
    startStationsSet = startStationsSet[:-1] # removes nulls
    usesPerStation = np.zeros(len(startStationsSet))
    for i in range(len(usesPerStation)):
        for station in startStations:
            if startStationsSet[i] == station:
                usesPerStation[i] += 1
    topTenInds = usesPerStation.argsort()[-10:][::-1] # finds the indicies of the top ten used start stations
    topTenStarts = np.zeros(len(topTenInds))
    topTenUsages = np.zeros(len(topTenInds))

    for i in range(len(topTenInds)):
        topTenStarts[i] = startStationsSet[topTenInds[i]]
    for i in range(len(topTenInds)):
        topTenUsages[i] = usesPerStation[topTenInds[i]]

    plt.figure(figsize=(11,7))
    sns.set_style("ticks")
    sns.barplot(x=topTenStarts, y=topTenUsages)
    plt.xlabel("Starting Station ID")
    plt.ylabel("Number of Uses")
    plt.tight_layout()
    plt.title("Top 10 Most Popular Start Stations")
    plt.savefig("static/graphs/popularStartStation.png", bbox_inches="tight", format="png")

    endStations = dataFrame['Ending Station ID']
    endStations = endStations.astype(str)
    endStationsSet = sorted(set(endStations)) # get a sigle list of all the end stations
    endStationsSet = endStationsSet[:-1] # removes nulls
    usesPerStation = np.zeros(len(endStationsSet))
    for i in range(len(usesPerStation)):
        for station in endStations:
            if endStationsSet[i] == station:
                usesPerStation[i] += 1
    topTenInds = usesPerStation.argsort()[-10:][::-1] # find the indicies of the top ten used end stations
    topTenEnds = np.zeros(len(topTenInds))
    topTenUsages = np.zeros(len(topTenInds))

    for i in range(len(topTenInds)):
        topTenEnds[i] = endStationsSet[topTenInds[i]]
    for i in range(len(topTenInds)):
        topTenUsages[i] = usesPerStation[topTenInds[i]]

    plt.figure(figsize=(11,7))
    sns.set_style("ticks")
    sns.barplot(x=topTenEnds, y=topTenUsages)
    plt.xlabel("Ending Station ID")
    plt.ylabel("Number of Uses")
    plt.tight_layout()
    plt.title("Top 10 Most Popular End Stations")
    plt.savefig("static/graphs/popularEndStations.png", bbox_inches="tight", format="png",)

def bike_sharing_breakdown():
    dataFrame = pd.read_csv('./data/metro-bike-share-trip-data.csv', low_memory=False)
    totalRides = len(dataFrame.index)
    ridePassTypes = dataFrame['Passholder Type']
    ridePassTypes = ridePassTypes.astype(str)

    regularPassTypes = ridePassTypes[ridePassTypes != "Walk-up"]
    regularRiders = len(regularPassTypes.index)
    walkUps = ridePassTypes[ridePassTypes == "Walk-up"]
    walkUpRides = len(walkUps)

    monthlyPasses = ridePassTypes[ridePassTypes == "Monthly Pass"]
    monthlyPassRides = len(monthlyPasses)
    flexPasses = ridePassTypes[ridePassTypes == "Flex Pass"]
    flexPassRides = len(flexPasses)
    staffAnnuals = ridePassTypes[ridePassTypes == "Staff Annual"]
    staffAnnualRides = len(staffAnnuals)

    bikeDistrosTot = [walkUpRides, monthlyPassRides, flexPassRides, staffAnnualRides]
    bikeDistrosPerc = np.zeros(len(bikeDistrosTot))
    for i in range(len(bikeDistrosTot)):
        bikeDistrosPerc[i] = round((bikeDistrosTot[i]/totalRides)*100, 3)
    #print(sorted(set(ridePassTypes)))
    labels = ["Walk-Ups", "Monthly Passes", "Flex Passes", "Staff Annuals"]
    for i in range(len(bikeDistrosTot)):
        labels[i] += ": " + str(bikeDistrosPerc[i]) + "%"

    # Donut plot to show the pass distribution
    plt.figure(figsize=(11,7))
    plt.subplot(121)
    plt.pie(bikeDistrosPerc, startangle=90)
    donutPlotCircle = plt.Circle((0,0), 0.7, color="white")
    plt.gcf().gca().add_artist(donutPlotCircle)
    plt.legend(labels)
    plt.title("Ride Pass Distribution")

    # Barplot to show the number of regular vs irregular users of the bike sharing system
    plt.subplot(122)
    sns.set_style("ticks")
    sns.barplot(x=["Irregular User Rides", "Regular User Rides"], y=[walkUpRides, regularRiders])
    plt.xlabel("Type of User")
    plt.ylabel("Number of total rides")
    plt.tight_layout();
    plt.title("Users of Bike Sharing")

    plt.savefig("static/graphs/passBreakdown.png", bbox_inches="tight", format="png")

def avg_dist_trav():
    def coordinate_dist(lat1, long1, lat2, long2): # Calculated using Haversine Formula
        if ((lat1==lat2) and (long1 == long2)):
            return 0
        R = 6371 # radius of earth in Kilometers
        lat1 = radians(lat1)
        long1 = radians(long1)
        lat2 = radians(lat2)
        long2 = radians(long2)
        deltaLat = lat2 - lat1
        deltaLong = long2 - long1
        a = sin(deltaLat/2)**2 + cos(lat1) * cos(lat2) * sin(deltaLong/2)**2
        c = 2 * asin(sqrt(a))
        km = R * c
        converstionFactor = 0.0621371 # go from kilometers to miles
        miles = km *converstionFactor
        return miles

    dataFrame = pd.read_csv('./data/metro-bike-share-trip-data.csv', low_memory=False)
    print(dataFrame)
    colsOfInterest = ["Duration", "Starting Station Latitude", "Starting Station Longitude",
                      "Ending Station Latitude", "Ending Station Longitude", "Trip Route Category"]
    #dataFrame = dataFrame.astype(str)
    #print(dataFrame)
    dataFrame = dataFrame[colsOfInterest]
    print(dataFrame)

def generate_graphs():
    popular_stations()
    bike_sharing_breakdown()
    avg_dist_trav();

if __name__ == '__main__':
    generate_graphs()
    #print(plt.style.available)

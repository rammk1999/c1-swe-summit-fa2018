# creates all the visuals for the webapp

from math import radians, cos, sin, asin, sqrt, isnan
import numpy as np
import pandas as pd
import scipy as sp
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
    sns.barplot(x=["Irregular Riders", "Regular Riders"], y=[walkUpRides, regularRiders])
    plt.xlabel("Type of User")
    plt.ylabel("Number of total rides")
    plt.tight_layout();
    plt.title("Irregular vs Regular Riders")

    plt.savefig("static/graphs/passBreakdown.png", bbox_inches="tight", format="png")

def avg_dist_trav():
    #Haversine formula for finding distances between two gps coordinates
    def coordinate_dist(lat1, long1, lat2, long2):
        if isnan(lat1) or isnan(long1) or isnan(lat2) or isnan(long2):
            return 0
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
        converstionFactor = 0.621371 # go from kilometers to miles
        miles = km *converstionFactor
        return miles

    dataFrame = pd.read_csv('./data/metro-bike-share-trip-data.csv', low_memory=False, encoding="utf-8")
    colsOfInterest = ["Duration", "Starting Station Latitude", "Starting Station Longitude",
                      "Ending Station Latitude", "Ending Station Longitude", "Trip Route Category"]
    dataFrame = dataFrame.astype(str)
    dataFrame = dataFrame[colsOfInterest]
    oneWays = dataFrame.loc[dataFrame["Trip Route Category"] == "One Way"]
    roundTrips = dataFrame.loc[dataFrame["Trip Route Category"] == "Round Trip"]

    ###################################################################################
    # this section finds the average speed if the distance is unknown (aka Round-Trips)
    avgSpeedsDf = oneWays[np.abs(sp.stats.zscore(oneWays["Duration"].astype(float))) < 3]
    # avgSpeedsDf is a data frame where all trips with a duration over 3 standard deviations from the mean
    # duration have been omitted for getting a more accurate average speed calculation
    avgBikerSpeed = 0 # Miles/Hr
    oneWayDistsTrimmed = [] # Converted to Miles
    oneWayTimesTrimmed = [] # Coverted to Hours
    for row in avgSpeedsDf.itertuples():
        tripDist = coordinate_dist(float(row[2]), float(row[3]), float(row[4]), float(row[5]))
        oneWayDistsTrimmed.append(tripDist)
        timeInSecs = float(row[1])
        timeInHours = ((timeInSecs / 60)/60)
        oneWayTimesTrimmed.append(timeInHours)
    oneWayTimesTrimmed = np.array(oneWayTimesTrimmed)
    oneWayDistsTrimmed = np.array(oneWayDistsTrimmed)
    oneWaySpeeds = oneWayDistsTrimmed/oneWayTimesTrimmed
    avgBikerSpeed = np.mean(oneWaySpeeds)
    ####################################################################################

    oneWayDists = [] # In miles
    for row in oneWays.itertuples():
        tripDist = coordinate_dist(float(row[2]), float(row[3]), float(row[4]), float(row[5]))
        oneWayDists.append(tripDist)
    oneWayDists = np.array(oneWayDists)
    roundTripDists = []
    for row in roundTrips.itertuples():
        timeInSecs = float(row[1])
        timeInHours = ((timeInSecs/ 60)/60)
        roundTripDists.append(timeInHours * avgBikerSpeed)
    roundTripDists = np.array(roundTripDists)
    totalDists = sum(oneWayDists) + sum(roundTripDists)
    totalRides = len(oneWayDists) + len(roundTripDists)
    avgDistancePerRide = totalDists/totalRides
    return avgDistancePerRide, avgBikerSpeed

def seasonal_effects():
    dataFrame = pd.read_csv("./data/metro-bike-share-trip-data.csv", low_memory=False)
    colsOfInterest = dataFrame.columns[:-2].astype(str)
    july = []
    august = []
    sept = []
    oct = []
    nov = []
    dec = []
    jan = []
    feb = []
    march = []
    for row in dataFrame.itertuples():
        monthNum = int(row[3][5:7])
        if monthNum == 7:
            july.append(row[1:-2])
        if monthNum == 8:
            august.append(row[1:-2])
        if monthNum == 9:
            sept.append(row[1:-2])
        if monthNum == 10:
            oct.append(row[1:-2])
        if monthNum == 11:
            nov.append(row[1:-2])
        if monthNum == 12:
            dec.append(row[1:-2])
        if monthNum == 1:
            jan.append(row[1:-2])
        if monthNum == 2:
            feb.append(row[1:-2])
        if monthNum == 3:
            march.append(row[1:-2])

    julyDf = pd.DataFrame(data=july, columns=colsOfInterest)
    augDf = pd.DataFrame(data=august, columns=colsOfInterest)
    septDf = pd.DataFrame(data=sept, columns=colsOfInterest)
    octDf = pd.DataFrame(data=oct, columns=colsOfInterest)
    novDf = pd.DataFrame(data=nov, columns=colsOfInterest)
    decDf = pd.DataFrame(data=dec, columns=colsOfInterest)
    janDf = pd.DataFrame(data=jan, columns=colsOfInterest)
    febDf = pd.DataFrame(data=feb, columns=colsOfInterest)
    marchDf = pd.DataFrame(data=march, columns=colsOfInterest)

    monthsDfs = [julyDf, augDf, septDf, octDf, novDf, decDf, janDf, febDf, marchDf]
    regularsTrend = []
    irregularsTrend = []
    totalTrend = []
    for dataFrame in monthsDfs:
        monthPasses = dataFrame['Passholder Type'].astype(str)
        regulars = len(monthPasses[monthPasses != "Walk-up"].index)
        irregulars = len(monthPasses[monthPasses == "Walk-up"].index)
        monthRides = regulars + irregulars
        regularsTrend.append(regulars)
        irregularsTrend.append(irregulars)
        totalTrend.append(monthRides)
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    labels = ["Regular Users", "Irregular Users", "Total Users"]
    plt.figure(figsize=(11,7))
    sns.lineplot(x=months, y=regularsTrend)
    sns.lineplot(x=months, y=irregularsTrend)
    sns.lineplot(x=months, y=totalTrend)
    plt.xlabel("Month Number (July to March)")
    plt.ylabel("Total Rides")
    plt.title("Monthly Impact on number of Users")
    plt.tight_layout()
    plt.legend(labels)
    plt.savefig("./static/graphs/seasonEffects.png", bbox_inches = "tight", format= "png")


def largest_net_change():
    dataFrame = pd.read_csv('./data/metro-bike-share-trip-data.csv', low_memory=False)
    colsOfInterest = ['Start Time', 'Starting Station ID', 'Ending Station ID']
    stations = dataFrame[colsOfInterest].astype(str)
    startStations = stations["Starting Station ID"]
    startStationsSet = sorted(set(startStations))[:-1]
    startUses = np.zeros(len(stations))
    endStations = stations["Ending Station ID"]
    endStationsSet = sorted(set(endStations))[:-1]
    endUses = np.zeros(len(stations))
    for i in range(len(startStationsSet)):
        for station in startStations:
            if startStationsSet[i] == station:
                startUses[i] += 1
    for i in range(len(endStationsSet)):
        for station in endStations:
            if endStationsSet[i] == station:
                endUses[i] += 1
    netUses = startUses - endUses
    topTenInds = netUses.argsort()[:10] # top 10 biggest negative net differences
    topLossStations = np.zeros(len(topTenInds))
    netLosses = np.zeros(len(topTenInds))
    for i in range(len(topTenInds)):
        topLossStations[i] = startStationsSet[topTenInds[i]]
        netLosses[i] = netUses[topTenInds[i]]
    netLosses = abs(netLosses)

    # calculate the duration of the data collection to find the avgerage loss per day
    dates = dataFrame["Start Time"]
    firstDay = dates[0]
    lastDay = dates[dates.last_valid_index()]
    d1 = filter_time(firstDay)
    d2 = filter_time(lastDay)
    totalDays = abs((d2-d1).days)
    netLosses /= totalDays
    plt.figure(figsize=(11,7))
    sns.set_style("ticks")
    sns.barplot(x=topLossStations, y=netLosses)
    plt.xlabel("Station ID")
    plt.ylabel("Average Bikes Moved per Day")
    plt.tight_layout()
    plt.title("Stations with the Most Displaced Bikes per Day")
    plt.savefig("static/graphs/netChange.png", bbox_inches="tight", format="png")
def generate_graphs():
    #popular_stations()
    #print("Generated popular station graphs")
    #bike_sharing_breakdown()
    #print("Generated bike sharing breakdown graphs")
    #avg_dist_trav();
    #print("Found average distance travelled")
    #seasonal_effects()
    #print("Discovered seasonal effects")
    largest_net_change()

if __name__ == '__main__':
    generate_graphs()
    #print(plt.style.available)

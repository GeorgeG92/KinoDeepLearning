import pandas as pd
import numpy as np
import os
from numpy import random as nprandom
import random
import time

winCols = ['WinNo1', 'WinNo2', 'WinNo3', 'WinNo4', 'WinNo5', 'WinNo6', 'WinNo7', 'WinNo8', 'WinNo9', 'WinNo10',
			 'WinNo11', 'WinNo12', 'WinNo13', 'WinNo14', 'WinNo15', 'WinNo16', 'WinNo17', 'WinNo18', 'WinNo19',
			 'WinNo20']

featureCols = ['id', 'date', 'time']

daysofweek = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
monthsofyear = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
				'September', 'October', 'November', 'December']

renameDict = {'Αριθμός Κλήρωσης':'id', 'Ημ/νία Κλήρωσης':'date',
							 'Ώρα Κλήρωσης':'time', '1ος ':'WinNo1', '2ος ':'WinNo2', '3ος ':'WinNo3', '4ος ':'WinNo4',
							 '5ος ':'WinNo5', '6ος ':'WinNo6', '7ος ':'WinNo7', '8ος ':'WinNo8', '9ος ':'WinNo9', 
							  '10ος ':'WinNo10', '11ος ':'WinNo11', '12ος ':'WinNo12', '13ος ':'WinNo13', '14ος ':'WinNo14',
							  '15ος ':'WinNo15', '16ος ':'WinNo16', '17ος ':'WinNo17', '18ος ':'WinNo18', '19ος ':'WinNo19',
							  '20ος ':'WinNo20'}

primenumbers = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

sinDoMDict28 = {}
for i in range(28):
	sinDoMDict28[i] = np.sin((2*np.pi*i)/28)
cosDoMDict28 = {}
for i in range(28):
	cosDoMDict28[i] = np.cos((2*np.pi*i)/28)
sinDoMDict30 = {}
for i in range(30):
	sinDoMDict30[i] = np.sin((2*np.pi*i)/30)
cosDoMDict30 = {}
for i in range(30):
	cosDoMDict30[i] = np.cos((2*np.pi*i)/30)
sinDoMDict31 = {}
for i in range(28):
	sinDoMDict31[i] = np.sin((2*np.pi*i)/31)
cosDoMDict31 = {}
for i in range(28):
	cosDoMDict31[i] = np.cos((2*np.pi*i)/31)

sindowDict = {}
for idx, day in enumerate(daysofweek):
	sindowDict[day]=np.sin((2*np.pi*idx)/len(daysofweek))

cosdowDict = {}
for idx, day in enumerate(daysofweek):
	cosdowDict[day]=np.cos((2*np.pi*idx)/len(daysofweek))

sinmoyDict = {}
for idx, month in enumerate(monthsofyear):
	sinmoyDict[month]=np.sin((2*np.pi*idx)/len(monthsofyear))

cosmoyDict = {}
for idx, month in enumerate(monthsofyear):
	cosmoyDict[month]=np.cos((2*np.pi*idx)/len(monthsofyear))

def addTransformations(df):
	"""
	Adds exp and squared transformations to all features
	Args:
		df: A dataframe containing features
	Returns:
		A dataframe with all additional transformed features
	"""
	relCols = [col for col in featuresDf.columns if col not in winCols]
	for col in relCols:
		df['e_'+col] = np.round(np.exp(df[col]),7)
		df['sq_'+col] = np.round(np.square(df[col]),7)     # when x in [-1,+1], log function is -inf
	return df


def processIds(df):
	"""
	Performs feature engineering on  ['id'] using prime number modulos
	Args:
		df: A dataframe containing the 'id' column
	Returns:
		A dataframe with all calculated id features
	"""
	for primeno in primenumbers:
		df['sinId%'+str(primeno)] = np.round(np.sin(2*np.pi*(df['id']%primeno)/primeno),7)
		df['cosId%'+str(primeno)] = np.round(np.cos(2*np.pi*(df['id']%primeno)/primeno),7)

		df['e_sinId%'+str(primeno)] = np.round(np.exp(df['sinId%'+str(primeno)]),7)
		df['e_cosId%'+str(primeno)] = np.round(np.exp(df['cosId%'+str(primeno)]),7)

		df['sq_sinId%'+str(primeno)] = np.round(np.square(df['sinId%'+str(primeno)]),7)
		df['sq_cosId%'+str(primeno)] = np.round(np.square(df['cosId%'+str(primeno)]),7)
	return df

def processFeatures(featuresDf):
	"""
	Performs feature engineering on  ['id', 'date', 'time'] using
		periodic features and various transformations
	Args:
		featuresDf: A dataframe containing ['id', 'date', 'time']
	Returns:
		A dataframe with all calculated features
	"""
	df = featuresDf.copy()
	df['date'] = pd.to_datetime(df['date'])
	df['time'] = pd.to_datetime(df['time'])
	df['Hour'] = df['time'].dt.hour
	df['Minutes'] = df['time'].dt.minute
	df['Day'] = df['date'].dt.day
	df['Month'] = df['date'].dt.month
	df['Year'] = df['date'].dt.year

	df['DayT'] = df['date'].dt.day_name()
	df['MonthT'] = df['date'].dt.month_name()
		
	# Hour of day
	df['sinHod'] = np.round(np.sin((2*np.pi*(df['Hour'])/24)),7)
	df['cosHod'] = np.round(np.cos((2*np.pi*(df['Hour'])/24)),7)

	# Hour of half day (12h)
	df['sinHohd'] = np.round(np.sin((2*np.pi*(df['Hour'])/12)),7)
	df['cosHohd'] = np.round(np.cos((2*np.pi*(df['Hour'])/12)),7)

	# Minute of Hour
	df['sinMoH'] = np.round(np.sin((2*np.pi*(df['Minutes'])/60)),7)
	df['cosMoH'] = np.round(np.cos((2*np.pi*(df['Minutes'])/60)),7)

	# Minute of day
	df['sinMoD'] = np.round(np.sin((2*np.pi*(df['Hour']*60+featuresDf['Minutes'])/1440)),7)
	df['cosMoD'] = np.round(np.cos((2*np.pi*(df['Hour']*60+featuresDf['Minutes'])/1440)),7)

	# DayName of week
	df['sinDow'] = np.round(df['DayT'].map(sindowDict),7)              # angle represents periodic sin of Day of week!
	df['cosDow'] = np.round(df['DayT'].map(cosdowDict),7)


	# Day of Month
	df['sinDoM28'] = np.round(df['Month'].map(sinDoMDict28),7)
	df['cosDoM28'] = np.round(df['Month'].map(cosDoMDict28),7)
	df['sinDoM30'] = np.round(df['Month'].map(sinDoMDict30),7)
	df['cosDoM30'] = np.round(df['Month'].map(cosDoMDict30),7)
	df['sinDoM31'] = np.round(df['Month'].map(sinDoMDict31),7)
	df['cosDoM31'] = np.round(df['Month'].map(cosDoMDict31),7)

	# Month of Year
	df['sinMoy'] = df['MonthT'].map(sinmoyDict)
	df['cosMoy'] = df['MonthT'].map(cosmoyDict)

	# Year of Decade
	df['sinYoD'] = np.round(np.sin((2*np.pi*(df['Year']%10)/10)),7)
	df['cosYoD'] = np.round(np.cos((2*np.pi*(df['Year']%10)/10)),7)
	df = df.drop(columns=['date', 'time', 'Hour', 'Minutes', 'Day', 'Month', 'DayT', 'MonthT', 'Year'])
	
	df = addTransformations(df)
	df = processIds(df)
	return df

def processTargets(targetsDf):
	"""
	Preprocess winning numbers: put them in ascending order
	Args:
		targetsDf: A dataframe containing targets (winning numbers)
	"""
	winNumbers = targetsDf.values
	winNumbers.sort(axis=1)                 # Sort winning numbers (asc)
	winNumbersDf = pd.DataFrame(winNumbers, targetsDf.index, winCols)
	return winNumbersDf



def dataLoader(filepath, featurepath):
	"""
	Loads data, performs feature engineering, persists resulting file
		and returns the resulting dataframe
	Args:
		filepath: the path to the original data file
		featurepath: path to the features file 
	"""
	featuresFile = os.path.join(featurepath, "features.csv")
	if os.path.exists(featuresFile):
		print("\tFeatures file exists!")
		totalDf = pd.read_csv(featuresFile)
	else:
		print("\tFeatures file doesn't exist, loading data...")
		drawsDf = pd.read_csv(filepath)
		drawsDf = drawsDf.rename(columns=renameDict)
		winNumbersDf = processTargets(drawsDf[winCols])
		featuresDf = processFeatures(drawsDf[featureCols])
		del drawsDf
		# Merge the two dataframes and set 'id ' as first column
		totaldf = winNumbersDf.join(featuresDf, how='outer')
		idCol = totaldf.pop('id')
		totaldf.insert(0, 'id', idCol)
	return totalDf
import numpy as np
import tensorflow as tf

class model():
	def __init__(self, args, dataDf):
		self.train = args.train
		self.batchsize = args.batchsize
		self.lookback = args.lookback
		self.modelpath = args.modelpath
		self.overwrite = args.overwrite
		self.output = args.output
		self.winNos = 20
		self.contFDict = {}               # dictionary of continuous features
		self.seqFDict = {}                # dictionary of seq features (past winning numbers)
		self.winNumbersTDict = {}         # dictionary of targets

		if self.train:
			print("Preparing to train")
			self.prepareDictionaries(dataDf)
			self.defineModelArchitecture()
			self.trainModel()
		else:
			print("Loading model for predictions")
			# Load model

	def defineModelArchitecture(self):
		return 0

	def getPastWinningNumbers(self, datavalues, drawId): 
		"""Processes arguments and performs validity checks
		Args:
			datavalues: a numpy array of all past winning numbers
			drawId: the id of the draw we want past winning numbers for
		Returns:
			A slice of the input that contains lookback draws of winning numbers,
			 e.g for draw 12 and lookback 5 it contains a (5,20) matrix
		"""
		return datavalues[int(drawId-self.lookback):drawId, 1:13]

	def prepareDictionaries(self, data):
		""" Prepares the dictionaries that will be used during batch prep
		Args:
			data: 
		"""
		print("\tCreating dictionaries for batchPrep")
		datavalues = data.values
		for example in [x for x in data.iloc[self.lookback:].values]:
			drawId = int(example[0])
			winNumbers = example[1:self.winNos+1]
			self.seqFDict[drawId] = self.getPastWinningNumbers(datavalues,drawId)          # get previous 'lookback' times winning numbers
			self.contFDict[drawId] = example[self.winNos+1:]                       			   # continuous features
			self.winNumbersTDict[drawId] = winNumbers                   				   # the targets (12 numbers)
		return 0

	def trainModel(self):
		return 0

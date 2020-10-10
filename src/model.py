import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import random
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

class model():
	def __init__(self, args, dataDf):
		self.train = args.train
		self.batchsize = int(args.batchsize)
		self.batchesNo = 100
		self.lookback = args.lookback
		self.modelpath = args.modelpath
		self.overwrite = args.overwrite
		self.output = args.output
		self.winNos = 20
		self.contFDict = {}               # dictionary of continuous features
		self.seqFDict = {}                # dictionary of seq features (past winning numbers)
		self.winNumbersTDict = {}         # dictionary of targets

		self.prepareData(dataDf)
		if self.train:
			print("Preparing to train")
			self.initWeightsAndBiases()
			self.defineModelArchitecture()
			self.trainModel()
		else:
			print("Loading model for predictions")
			# Load model

	def absLoss(self, targets, predictions):
		""" Mean Absolute loss function for the TF graph
		Args:
			targets: the ground truth passed to the model
			predictions: the output of the model
		Returns:
			MAE metric
		"""
		return tf.reduce_mean(tf.math.abs(targets - predictions))

	def fcLayer(self, x, w, b, activation=None):
		""" Implementation of a fully connected/dense TF layer
		Args:
			x: the input of the layer
			w: the corresponding weights 
			b: the bias term
			activation: the activation function to be applied in the end
		Returns:
			The result of the computation
		"""
		result = tf.matmul(x, w) + b
		if activation == 'relu':
			result = tf.nn.leaky_relu(result)         # normal will cut off most values
		if activation == 'softmax':
			result = tf.nn.softmax(result,axis=1)
		return result

	def batchPrep(self):
		""" Prepares batches for training using prebuilt dictionaries
		Returns:
			2-D numpy arrays for continuous/sequence features and targets respectively
		"""
		batchDrawIds = random.sample(self.trainDrawIds, self.batchsize)
		contFeatures = np.array([self.contFDict[x] for x in batchDrawIds])
		seqFeatures = np.array([self.seqFDict[x] for x in batchDrawIds])
		targets = np.array([self.winNumbersTDict[x] for x in batchDrawIds])
		return contFeatures, seqFeatures, targets

	def getPastWinningNumbers(self, datavalues, drawId): 
		"""Processes arguments and performs validity checks
		Args:
			datavalues: a numpy array of all past winning numbers
			drawId: the id of the draw we want past winning numbers for
		Returns:
			A slice of the input that contains lookback draws of winning numbers,
			 e.g for draw 12 and lookback 5 it contains a (5,20) matrix
		"""
		return datavalues[int(drawId-self.lookback):drawId, 1:self.winNos+1]

	def prepareData(self, data):
		""" Prepares the dictionaries that will be used during batch prep
			and defined train, eval and test examples
		Args:
			data: 
		"""
		print("\tCreating dictionaries for batchPrep")
		datavalues = data.values
		for example in [x for x in data.iloc[self.lookback:].values]:
			drawId = int(example[0])
			self.seqFDict[drawId] = self.getPastWinningNumbers(datavalues,drawId)          # get previous 'lookback' times winning numbers
			self.contFDict[drawId] = example[self.winNos+1:]                       		   # continuous features
			self.winNumbersTDict[drawId] = example[1:self.winNos+1]                   	   # the targets (12 numbers)
		
		self.inputSize = self.contFDict[drawId].shape[0]
		self.inputSeqSize = self.seqFDict[drawId].shape
		self.targetSize = self.winNumbersTDict[drawId].shape[0]

		# Train/eval/test split
		trainDf , testDf = train_test_split(data.iloc[self.lookback:], train_size=0.8, test_size=0.2)   # 80-10-10 split
		testDf, evalDf = train_test_split(testDf, train_size=0.5, test_size=0.5)
		self.trainDrawIds = set(trainDf['id'])
		self.evalDrawIds = set(evalDf['id'])
		self.testDrawIds = set(testDf['id'])

		# TF config and saver
		self.config = tf.ConfigProto()
		self.config.gpu_options.allow_growth = True
		self.config.log_device_placement = False
		self.config.allow_soft_placement = True
		return 0


	def initWeightsAndBiases(self):
		""" Initializes the trainable parameters of the model
		Returns:
			Success Code 
		"""
		self.biases = {
			'b1': tf.get_variable('b1', shape=(256,), initializer = tf.random_normal_initializer(0, 0.05)),
			'b2': tf.get_variable('b2', shape=(128,), initializer = tf.random_normal_initializer(0, 0.05)),
			'b3': tf.get_variable('b3', shape=(64,), initializer = tf.random_normal_initializer(0, 0.05)),
			'b4': tf.get_variable('b4', shape=(self.targetSize,), initializer = tf.random_normal_initializer(0, 0.05))
		}

		self.weights = {
			'w1': tf.get_variable('w1', shape=(self.inputSize, 256), initializer = tf.random_normal_initializer(0, 0.05)),
			'w2': tf.get_variable('w2', shape=(256, 128), initializer = tf.random_normal_initializer(0, 0.05)),
			'w3': tf.get_variable('w3', shape=(128, 64), initializer = tf.random_normal_initializer(0, 0.05)),
			'w4': tf.get_variable('w4', shape=(64, self.targetSize), initializer = tf.random_normal_initializer(0, 0.05))
		}
		return 0

	def defineModelArchitecture(self):
		""" Defines the model architecture: the input placeholders and the operations within the graph
		Returns:
			Success Code
		"""

		# Graph operations
		self.featuresInput = tf.placeholder(name="featuresInput", dtype=tf.float32, shape=(None, self.inputSize))
		#seqInput = ...
		self.targetsOutput = tf.placeholder(name="targetsOutput", dtype=tf.float32, shape=(None, self.targetSize))

		with tf.variable_scope("contFeatures"):
			self.x1 = self.fcLayer(self.featuresInput, self.weights['w1'], self.biases['b1'], 'relu')
			self.x2 = self.fcLayer(self.x1, self.weights['w2'], self.biases['b2'], 'relu')
			self.x3 = self.fcLayer(self.x2, self.weights['w3'], self.biases['b3'], 'relu')
			self.output = self.fcLayer(self.x3, self.weights['w4'], self.biases['b4']) 
		# with tf.variable_scope("seqFeatures"):
		#     y1 = 
			#y2 = 
		# Concatenate
		# Add a few more fcLayers
		#with tf.variable_scope("jointFeatures"):

		self.loss = self.absLoss(self.targetsOutput, self.output)
		self.optim = tf.train.AdamOptimizer(0.002, 0.5)
		self.train = self.optim.minimize(self.loss)
		return 0


	def runInference(self, runType, sess):
		""" Used to run model on eval/test data and compute appropriate metrics
		Args:
			runType: 'eval' or 'test'
			sess: The TF session passed from the train function for inference
		"""
		if runType=='eval':
			contFeatures = np.array([self.contFDict[x] for x in self.evalDrawIds])
			seqFeatures = np.array([self.seqFDict[x] for x in self.evalDrawIds])
			targets = np.array([self.winNumbersTDict[x] for x in self.evalDrawIds])
		elif runType=='test':
			contFeatures = np.array([self.contFDict[x] for x in self.testDrawIds])
			seqFeatures = np.array([self.seqFDict[x] for x in self.testDrawIds])
			targets = np.array([self.winNumbersTDict[x] for x in self.testDrawIds])
		else:
			raise Exception('runInference() runType should be in "eval" or "test", '+str(runType)+" provided")
		
		fd = {self.featuresInput: contFeatures, self.targetsOutput: targets}
		_, l, out = sess.run(fetches=[self.train, self.loss, self.output], feed_dict=fd)
		predictions = np.round(out)
		sqLoss = mean_absolute_error(targets, predictions)# ((targets - predictions)**2).mean()        # mean average error
		if runType=='eval':
			print("\tEval MAE: ", round(sqLoss,3))
		else:
			print("\tTest MAE: ", round(sqLoss,3))
			#processResults(targets, predictions)
		return predictions

	def trainModel(self):
		"""
		Initializes a TF session, performs training and predicts on eval/test data
		Returns:
			Success Code
		"""
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			self.saver = tf.train.Saver()
			print("Training:")
			for i in range(self.batchesNo):
				contFeatures, seqFeatures, targets = self.batchPrep()
				fd = {self.featuresInput: contFeatures, self.targetsOutput: targets}
				_, l, out, xx1, fi = sess.run(fetches=[self.train, self.loss, self.output, self.x1, self.featuresInput], feed_dict=fd)
				if i%(self.batchesNo/10)==0:
					print("\tBatch ",i,": Train loss: ", l)
				if i>0 and i%100==0:
					self.runInference(runType='eval', sess=sess)
			print("Testing:")
			self.runInference(runType='test', sess=sess)
		return 0

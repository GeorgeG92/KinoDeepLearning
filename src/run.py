from argParser import argParser
from preprocess import dataLoader 
from multiheadedModel import model
#from regressionModel import model
import sys



if __name__ == "__main__":
    args = argParser(sys.argv)
    data = dataLoader(args.filepath, args.featurepath)
    model = model(args, data)
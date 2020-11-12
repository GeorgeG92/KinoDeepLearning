import sys
import argparse
import os

def argParser(argv):
    """Add command line arguments and parse user inputs.
    Args:
        argv: User input from the command line.
    Returns:
        An args object that contains parsed arguments.
    """
    # Creating a parser
    parser = argparse.ArgumentParser(description='My Model')
    
    parser.add_argument('-l', '--lookback', dest='lookback',
                         default=10,
                        help='How many draws to consider as LSTM input')

    parser.add_argument('-t', '--train', dest='train',
                        action='store_true', default=True,
                        help='Whether to train the model or use as is')

    parser.add_argument('-b', '--batchsize', dest='batchsize',
                        default='64',
                        help='Define training batchsize')

    parser.add_argument('-m', '--modelpath', dest='modelpath',
                        default=os.path.join('..', 'model'),
                        help='Path to tensorflow model')

    parser.add_argument('--filepath', dest='filepath',
                        default=os.path.join('..', 'data', 'allDraws.csv'),
                        help='Path to draws csv')

    parser.add_argument('--featurepath', dest='featurepath',
                        default=os.path.join('..', 'features'),
                        help='Path to features csv')
    
    parser.add_argument('-ow', '--overwrite', dest='overwrite',
                        action='store_true', default=True,
                        help='Overwrite existing model after training')

    parser.add_argument('-o', '--outputpath', dest='outputpath',
                        action='store_true', default=os.path.join('..', 'output'),
                        help='Path to model output/predictions output')

    args = parser.parse_args(argv[1:])
    argChecker(args)
    return args

def argChecker(args):
    """Processes arguments and performs validity checks
    Returns:
        Success Code
    """
    if not os.path.exists(args.outputpath):
        print('\tOutput directory not found, creating...')
        os.mkdir(args.outputpath)
    if not os.path.exists(args.modelpath):
        print('\tModel directory not found, creating...')
        os.mkdir(args.modelpath)
    if args.train==False:              
         assert 1, "Training is set to False but no compatible model found, terminating..."               # check if model is there!
    assert os.path.exists(args.filepath), "Data file not found, please check "+str(args.filepath)
    return 0


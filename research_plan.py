#!/usr/bin/python3

# set the epsilon value on about line 871
# set skip_build in main method
# n in main method

# command line argument processing
import sys;

# comand line parsing facility
import getopt;

# ascii digit characters
import string;

# make a copy
import copy;

# natural logarithm
import math;

# variance
import statistics;

# pandas dataframe
import pandas as pd;

# numpy array
import numpy as np;

# decision tree
from sklearn.tree import DecisionTreeClassifier; 

# dividing the dataframe into training/test data
from sklearn.model_selection import train_test_split;

# testing metrics
from sklearn import metrics;

# training/testing data from OASIS
import training as oas;

# utilities has a print_list method
import utilities;

# the menu driven interface has the functionality necessary
import sensitivity_epsilon as epsilon;

# interaction loop commands, descriptions, and index
#COMMAND_STRS = ["exit", 
#        "patient",
#        "sensitivity",
#        "machine",
#        "test",
#        "epsilon",
#        "write",
#        "load"];

#COMMAND_INTS = [0, 
#        1, 
#        2, 
#        3,
#        4,
#        5,
#        6,
#        7];

#COMMAND_DESCS = ["0 - Exit",
#        "1 - Compare Machine Result with Actual Patient",
#        "2 - Perform Sensitivity Analysis on Current Machine",
#        "3 - Select Another Machine",
#        "4 - Run Accuracy Test",
#        "5 - Set Epsilon on Copy of Current Machine",
#        "6 - Write A Machine to Disk",
#        "7 - Load A Machine from Disk"];

#COMMAND_FUNCS = [];

# the number of columns for the trained machine
#DATASET_COLUMNS = int();

# the number of y value columns
#DATASET_Y_VALUES = int();

# the various types of models that can be found
#DECISION_TREE = 0;
#NEURAL_NETWORK = 1;

# control debug printing
DEBUG = False;

#######################################
def main(argc, argv):

    # default size and proportion for this main program
    sz = 500;
    fraction = 0.8;
    ep = float();
    entropy = float();
    path = None;

    # take care that there is a compatible
    # machine to load
    skip_build = False;

    # command line options and the associated parameter if any
    sopt = "d:"; # "d:e"; -- fix getopt
    lopt = ["directory"]; # , "epsilon"]; -- fix getopt
    optval = None;
    trail = None;

    # default machine attributes
    names = list(); 

    if not skip_build:
        names = ["Decision Tree", "Noisy Decision Tree"];

    machines = list();
    Xs = list();
    Ys = list();
    datasets = list();
    eps = list();
    entropys = list();
    types = list();

    # test minimal usage; print usage message
    if argc == 1:

        usage(argv[0]);
        return 1;

    # there are arguments, parse them
    else:
        try:
            argarr = argv[1:];
            optval, trail = getopt.getopt(argarr, sopt, lopt);
            # print("optval: " + str(optval));
            # print("trail: " + str(trail));

        except getopt.GetoptError as err:
            print(err);
            usage(argv[0]);
            return 2;

        # for every one specified; take the last one given
        for op, ar in optval:

            #print("op = " + str(op) + "; ar = " + str(ar));

            if op in ("-d", "--directory"):

                path = ar;

    # permit extra arguments, but advise user that they are ignored
    if len(trail) > 0:

        print("Extra arguments are ignored: " + str(trail) + "...continuing.");

    if not skip_build:
        print("Building models from dataset in " + str(path) + " now..."); 

        # for each model mentioned in list: names
        for i in range(len(names)):

            ai_type = -1;
            add = False;

            # produce each model, if possible
            print("=== " + names[i] + " ===");

            if "Decision Tree" == names[i]:

                # run the build to create the base line dataset
                ep, entropy, mach, X, Y, P =\
                    epsilon.build_decision_tree(path, sz, fraction, epsilon.filled);
                ai_type = epsilon.DECISION_TREE;    
                add = True;

            elif "Noisy Decision Tree" == names[i]:

                # run the build to create the base line dataset
                ep, entropy, mach, X, Y, P =\
                    epsilon.build_decision_tree(path, sz, fraction, epsilon.noisy);
                ai_type = epsilon.DECISION_TREE;
                add = True;

            else:

                # report that the machine build is not ready
                print("The machine: " + name[i]\
                + "'s build routine is not ready.");

            if True == add:

                # update machine attributes
                machines.append(copy.deepcopy(mach));
                Xs.append(copy.deepcopy(X));
                Ys.append(copy.deepcopy(Y));
                datasets.append(copy.deepcopy(P));
                eps.append(copy.deepcopy(ep));
                entropys.append(copy.deepcopy(entropy));
                types.append(copy.deepcopy(ai_type));

    # skip building models
    else:
        print("Loading models from ./ (current working directory) as requested...");

    print("===========================================\n\n");

    # perform research plan
    # 1. Report accuracy of fill trained machine on filled data
    print("Research Question 1. Find accuracy for the baseline data:");
    accuracy_fill = epsilon.test_model(\
            machines[0], Xs[0], Ys[0], types[0]);

    # 2. Replace each all patients with a noise version; report accuracy 
    print("Research Question 2. Accuracy of baseline machine on noise data set:");
    accuracy_noise = epsilon.test_model(\
            machines[0], Xs[1], Ys[1], types[0]);

    # 3. Unbounded (partition) differential privacy
    print("Research Question 3. Sensitivity of accuracy in unbounded differential privacy:");
    delta_accuracy_unbounded =\
            epsilon.get_dt_unbounded_privacy(\
            machines[0], Xs[0], Ys[0], datasets[0]);

    # 4. Bounded (substitution) differential privacy
    print("Research Question 4. Sensitivity of accuracy in bounded differential privacy:");
    delta_accuracy_bounded =\
            epsilon.get_dt_bounded_privacy(\
            machines[0],\
            Xs[0],\
            Ys[0],\
            Xs[1],\
            Ys[1],\
            datasets[0],\
            datasets[1]);

    # 5. Report entropy measurements for each dataset
    print("Research Question 5: Entropy measurements:");

    # this should be placed in sensitivity_epsilon.py
    total = float();

    for i in range(len(datasets)):

        print("\n=== Machine: " + names[i] + " ===");

        print("Dimension\tEntropy");

        total = 0.0;

        # report for each dimension; accumulate total
        for j in range(len(oas.session_data.dimension)):

            print(oas.session_data.dimension[j]\
                + "\t\t\t" + str(entropys[i][j]));
            total += entropys[i][j];

        print("Entropy (Per Element): " +\
                str(total / len(datasets[i])) + "\n");

    print();


    return 0;

if "__main__" == __name__:

    result = main(len(sys.argv), sys.argv);
    exit(result);


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

# interaction loop commands, descriptions, and index
COMMAND_STRS = ["exit", 
        "patient",
        "sensitivity",
        "machine",
        "test",
        "epsilon",
        "entropy",
        "write",
        "load"];

COMMAND_INTS = [0, 
        1, 
        2, 
        3,
        4,
        5,
        6,
        7,
        8];

COMMAND_DESCS = ["0 - Exit",
        "1 - Compare Machine Result with Actual Patient",
        "2 - Perform Sensitivity Analysis on Current Machine",
        "3 - Select Another Machine",
        "4 - Run Accuracy Test",
        "5 - Set Epsilon on Copy of Current Machine",
        "6 - Get Entroy of Machine Dataset",
        "7 - Write A Machine to Disk",
        "8 - Load A Machine from Disk"];

COMMAND_FUNCS = [];

# the number of columns for the trained machine
DATASET_COLUMNS = int();

# the number of y value columns
DATASET_Y_VALUES = int();

# the various types of models that can be found
DECISION_TREE = 0;
NEURAL_NETWORK = 1;

# control debug printing
DEBUG = False;

#######################################
def debug_table_columns(table, cols):

    print("Length of data table (rows x columns): " + str(len(table)) + " x "\
            + str(len(table[0])));
    print("Length of column names list: " + str(len(cols)));

    return;

#######################################
# Only called on the training data. Not to be
# updated again until another machine is.
#
# determine largest parameter; updates:
# DATASET_COLUMNS
def set_trim(table, titles):

    global DATASET_COLUMNS;

    row_len = list();

    for r in table:

        if len(r) > 0:

            row_len.append(len(r));

    shorty1 = min(row_len);
    shorty2 = len(titles);

    # is the length of a row less than the number of columns
    if shorty1 < shorty2:

        DATASET_COLUMNS = shorty1;

    # is the number of columns less than the length of a row
    elif shorty1 > shorty2:

        DATASET_COLUMNS = shorty2;

    # the shortest row and the number of column titles were equal
    else:

        DATASET_COLUMNS = shorty1;

    # debug
    #print("Trim length: " + str(DATASET_COLUMNS));

    return;

########################################
# trim either the column titles or the table columns
# to the size DATASET_COLUMNS
def do_trim(table, titles):

    global DATASET_COLUMNS;

    # add/remove columns (0x00), from the end of the table's 
    # rows, until each row is DATASET_COLUMNS in length
    table = utilities.trim_2d_columns_zeros(table, DATASET_COLUMNS);
    
    # add/remove entries (px0, px1, etc.) from a list
    # until it is DATASET_COLUMNS in length
    titles = utilities.trim_list_nils(titles, DATASET_COLUMNS);

    return table, titles;

########################################
# return X,Y from a dataset
def prepXY(dataset, training_data = False):

    global DATASET_Y_VALUES;

    X = pd.DataFrame();
    Y = pd.DataFrame();

    # get the column names and the tabulated data
    table, cols = oas.patient_data.dataset_table(dataset);

    # if training, find the smallest that will be used to train the 
    # decision tree
    if training_data == True:
        
        set_trim(table, cols);

    # adjust table and column sizes
    table, cols = do_trim(table, cols);

    # fix the number of y value columns
    if training_data == True:

        DATASET_Y_VALUES = cols.index("px0");
    
    print("=== Trimmed Sizes ===");
    debug_table_columns(table, cols);
    print();

    # create a dataframe
    dataset_df = pd.DataFrame(data = table, columns = cols);

    # independent variables are the pixel data
    X = dataset_df[cols[DATASET_Y_VALUES:]];
   
    # dependent variables are the demographic data
    Y = dataset_df[cols[:DATASET_Y_VALUES]];

    return X, Y, dataset;

#######################################
# construct and return noisy X, Y dataframes from dataset
def noisy(path, dataset, training_data = False):

    # copy the original dataset
    dataset_noisy = copy.deepcopy(dataset);

    # the number of obfuscated fields
    no_obfuscations = int();

    # the epsilon selected by the user
    ep = float();

    # prompt the user for the desired epsilon level
    print("=== Epsilon Selection");
    ep = -1.0;

    while ep <= 0.0:

        if ep <= 0.0:
            print("Please enter an epsilon larger than 0.0.");

        try:
            ep = float(input(\
                "Please enter an epsilon value: "));
        except:
            print("The value entered could not be converted to a float.");

        if ep > 0.0:
            print("Using epsilon: " + str(ep));

    # perform obfuscation on 100% of the data set
    # using the laplace mechanism:
    # F(x) = f(x) + lap(s/e)
    # where s is the range of a data value 
    # within the set 
    # (extracted within the obfuscate function)
    # e is the epsilon value the user selected
    dataset_noisy = oas.obfuscate(\
            1.0,\
            ep,\
            dataset_noisy);

    # update noise files for posterity
    utilities.add_noise_files(path, dataset_noisy);

    # record the number of obfuscations
    no_obfuscations =\
            oas.session_data.dimension_obfuscation;

    # report the number of obfuscations
    oas.session_data.print_2column(\
            "Obfuscations",\
            oas.session_data.dimension_obfuscation);

    # reset the counters
    oas.session_data.reset_update_counters();

    X, Y, dataset =\
        prepXY(\
            dataset_noisy,\
            training_data = training_data);

    return ep, no_obfuscations, X, Y, dataset;

#######################################
# construct and return filled X, Y dataframes from dataset
# return:
# epsilon: always 0.0
# substitutions: the number of null values in the original
# dataset which were populated randomly from known ranges 
# of categorical values
# X: the independent values from the dataset (image data)
# Y: the dependent values (after populating, padding, and
# and flattening
# dataset: the dataset resulting from filling empty values
# and padding to a standard length
def filled(path, dataset, training_data = False):

    dataset_filled = copy.deepcopy(dataset);
    no_substitutions = int();

    # create fill files for posterity, if they do not already exist
    utilities.add_fill_files(path, dataset_filled);

    # record the number of substitutions for this dataset
    no_substitutions =\
            oas.session_data.dimension_substitution;

    # report the number of substitutions
    oas.session_data.print_2column(\
            "Substitutions",\
            oas.session_data.dimension_substitution);
    
    # remind user that the fill files are not replaced
    print("Note: If there were no substitions, the fill files already existed.");

    # reset the counters
    oas.session_data.reset_update_counters();

    X, Y, dataset = prepXY(\
            dataset_filled, \
            training_data = training_data);

    return 0.0, no_substitutions, X, Y, dataset;

#######################################
# construct the decision tree from the data path using patients
# number of patients and training_percent of them for training 
# the model
def build_decision_tree(path, patients, training_percent, which = filled):

    # create the classifier
    decision_tree = DecisionTreeClassifier();

    # the patient data
    dataset = oas.training_dataset(path, patients);

    # which dataset to use; uses filled by default
    epsilon, substitutions, X, Y, dataset =\
            which(path, dataset, training_data = True);

    # distinguish training and test data
    X_train, X_test, Y_train, Y_test =\
            train_test_split(X, Y, \
            test_size = 1.0 - training_percent,\
            random_state = 1);

    print("=== Fitting ===");

    # fit the classifier to the data
    decision_tree.fit(X_train, Y_train);

    return (epsilon, substitutions, decision_tree, X, Y, dataset);

#######################################
# test the decision tree model using X, Y 
# variables
# verbose will report intermediate results
#
# return a (Python) list depicting accuracy
# for each column of Y
def test_decision_tree(tree, X, Y, verbose = True):

    global DEBUG;

    if verbose:
        print("\n=== Test Decision Tree ===");

    # the result of the test
    average_accuracy = float();
    accuracy_list = list();

    # get the classifiers prediction for the x input data
    Y_pred = tree.predict(X);

    # copy/data type caste
    Y_p = pd.DataFrame(data = Y_pred, columns = Y.columns, copy = True);

    # loop variables
    accuracy = float();
    count = float();

    if DEBUG:
        print("Y = " + str(Y));
        print("Y_p = " + str(Y_p));
        print("len(Y) = " + str(len(Y)));
        print("len(Y.columns) = " + str(len(Y.columns)));

        prompt = input("Press return to continue...");

    # write the dependent variables to a log file
    timestamp = str(utilities.get_timestamp());
    utilities.write_square_matrix_with_titles(\
        Y, Y.columns, "Columns", "./matrix" + timestamp + ".txt");

    if verbose:
        print("\n=== Accuracy ===");
        print("Column\tAccuracy");

    # for every column
    for j in range(len(Y.columns)):

        # reset this column's count
        count = float();

        # for every row
        for i in range(len(Y)):

            # if the true value is equal to the predicted value
            if Y.iloc[i, j] == Y_p.iloc[i, j]:

                # increment the count
                count += 1.0;

        # normalize the count for the number of rows
        accuracy = float(count / float(len(Y)));

        # add this column's accuracy to the list of accuracy's
        accuracy_list.append(accuracy);

        if verbose:
            print(str(Y.columns[j]) + ":\t" + str(accuracy));

        if DEBUG:
            # debug
            print("accuracy of "\
                + str(Y.columns[j]) + " = " + str(accuracy));

    average_accuracy = sum(accuracy_list) / len(accuracy_list);

    if verbose:
        print("Average Accuracy %: " + str(average_accuracy * 100) + "%");
        print();

    return accuracy_list;

###################################################
# takes a machine, its inputs and output categories as
# well as the type of machine to produce a list
# of accuracies compared with the match in the outputs
def test_model(mach, X, Y, ai_type):

    global DECISION_TREE;
    global NEURAL_NETWORK;

    result = float();

    if DECISION_TREE == ai_type:

        result = test_decision_tree(mach, X, Y);

    elif NEURAL_NETWORK == ai_type:

        print("Neural network test routines are not implemented.");
        pass;

    else:

        print("method test_model: Unknown AI model.");

    return result;

###################################################
# prompt the user for the machine to use
def get_machine_command(nm):

    print("=== Machine Selection ===\n\n");

    print("Please select from the following:");

    valid = list();
    sel = -1;

    while sel == -1:

        for i in range(len(nm)):

            valid.append(i);

            print(str(i) + " -- " + nm[i]);

        print();

        print("Please enter your selection from " + str(valid) + ": ");

        try:
            sel = int(input());

        except:
            print("Your selection must be an integer....");

        if sel not in valid:
            print("Your selection is not in the list....");
            sel = -1;

    return nm[sel];

###################################################
# unbounded differential privacy measurements
# distance(dataset, dataset`) = 1
    # the patient that is removed makes the noise
    # compared to the accuracy for the original 
    # patient included in the set
#
# both dataset and dataset' are the flattened, tabular data
#
# returns:
# each delta(f) aka sensitivity from 'Programming DP' for each
# columns' accuracy
def get_dt_unbounded_privacy(mach, X, Y, dataset, verbose = True):

    # accuracy for each column after excluding an element
    # from X, Y
    accuracy_noisy = list();

    # list of largest differences in accuracy from the original for each column
    delta_f = list();

    # original accuracy per column from machine(X, Y)
    orig_acc = list();

    # the original length of patients
    len_dataset = len(dataset);

    # average accuracy
    mean_accuracy = float();
    
    if verbose:
        print("\n=== Sensitivity Analysis on Accuracy ===");
        print("=== Unbounded Differential Privacy ===");

    # get accuracy on original X, Y parameters
    orig_acc = test_decision_tree(mach, X, Y, False);
    
    if DEBUG:
        print("Original Accuracy:\n" + str(orig_acc));

    if verbose:
        print("Partitioning (One Less Element)....");

    # for each patient, leave them out once
    for i in range(len_dataset):

        # duplicate the original X, Y, to find maximal delta
        X_second = copy.deepcopy(X);
        Y_second = copy.deepcopy(Y);

        # create the X,Y partition with one different element
        # requires leaving one out
        X_second.drop([i], inplace = True);
        Y_second.drop([i], inplace = True);
            
        if verbose:
            print("\nExcluding Element at Index: " + str(i));
            print(str(dataset[i]));
            
        # calculate accuracy
        accuracy_noisy.append(\
                test_decision_tree(mach, X_second, Y_second, False)); 
        
        if DEBUG:
            print("Current Partition Accuracy: " + str(accuracy_noisy[i]));

    if verbose:
        print("===========================================\n\n");
        print("\n=== Sensitivity Summary ===");

        print("=== Original Accuracy (ASCII)");
        print("Column\tAccuracy");

        for i in range(len(Y.columns)):
            print(str(Y.columns[i]) + ":\t" + str(orig_acc[i]));

        mean_accuracy = statistics.mean(orig_acc);
    
        print("Average Accuracy: " + str(mean_accuracy));

    # find maximum difference of accuracy of each dimension
    # of the flattened dependent variable
    for i in range(len(orig_acc)):

        # suppose it is zero
        delta_f.append(0);

        # every partitions' measurement of column i
        for j in range(len(accuracy_noisy)):

            prime = abs(\
                orig_acc[i] - accuracy_noisy[j][i]);

            if prime > delta_f[i]:
            
                delta_f[i] = prime;
    
    if verbose:
        print("\n=== Sensitivity = delta(f) = max(abs(f(d) - f(d`))) =");
        print("Column\tSensitivity of Accuracy");

        for i in range(len(Y.columns)):

            print(str(Y.columns[i]) + ":\t" +\
                    str(delta_f[i]));

        print("L1 norm = " + str(sum(delta_f)));
        print("L2 norm = " + str(utilities.L2_norm(delta_f)));

        print("===========================================\n\n");

    return delta_f;

################################################
# calculate the bounded dp for each dimension
# of the dependent variable Y
def get_dt_bounded_privacy(machine, XF, YF, XN, YN, datasetF, datasetN, verbose = True):

    Xf = copy.deepcopy(XF); # data frames for current modification of data set
    Yf = copy.deepcopy(YF);

    Xn = copy.deepcopy(XN); # noise used to modify original data frame
    Yn = copy.deepcopy(YN);

    datasetf = copy.deepcopy(datasetF); # human readable versions of 
    datasetn = copy.deepcopy(datasetN); # 'original' and 'noisy' list of patients

    orig_acc = float();
    len_dataset = int();
    accuracy_noisy = list();

    delta_f = list();

    if verbose:
        print("\n=== Sensitivity Analysis on Accuracy ===");
        print("=== Bounded Differential Privacy ===");

    orig_acc = test_decision_tree(machine, Xf, Yf, False); 

    # the original length of the dataset
    len_dataset = len(datasetf);

    # accuracy at each iteration
    accuracy_noisy = list();

    # for each element in the original (filled) dataset
    for i in range(len(YF)):

        Xf = copy.deepcopy(XF);
        Yf = copy.deepcopy(YF);

        if verbose:
            print("Obfuscating patient at index: " + str(i));
            print("Before Obfuscation:\n" + str(datasetf[i]));
            print();
            print("After Obfuscation:\n" + str(datasetn[i]));

        # update X, Y

        if DEBUG:
            print("len(Yf.columns) = " + str(len(Yf.columns)));
            print("len(Xf.columns) = " + str(len(Xf.columns)));

            print("len(Yn.columns) = " + str(len(Yn.columns)));
            print("len(Xn.columns) = " + str(len(Xn.columns)));

        # update x, y rows for the replaced patient
        row = Yn.iloc[i:i + 1].to_numpy();

        for j in range(len(row[0])):
            Yf.iloc[i, j] = row[0][j];

        row = Xn.iloc[i:i + 1].to_numpy();
        
        for j in range(len(row[0])):
            Xf.iloc[i, j] = row[0][j];

        accuracy_noisy.append(\
                test_decision_tree(\
                    machine,\
                    Xf, Yf,\
                    verbose = False));

    if verbose:
        print("===========================================\n\n");

        print("=== Original Accuracy (ASCII)");
        print("Column\tAccuracy");

        for i in range(len(YF.columns)):
            print(str(YF.columns[i]) + ":\t" + str(orig_acc[i]));

        mean_accuracy = statistics.mean(orig_acc);
    
        print("Average Accuracy: " + str(mean_accuracy));

    # find maximum difference of accuracy of each dimension
    # of the flattened dependent variable
    # i.e. for every partitions' measurement of column i
    for i in range(len(orig_acc)):

        # suppose it is zero
        delta_f.append(0);

        # for every patient that was removed
        # a new accuracy list was added to
        # accuracy noisy
        for j in range(len(accuracy_noisy)):

            prime = abs(\
                orig_acc[i] - accuracy_noisy[j][i]);

            if prime > delta_f[i]:
            
                delta_f[i] = prime;
    
    if verbose:
        print("\n=== Sensitivity = delta(f) = max(abs(f(d) - f(d`))) =");
        print("Column\tSensitivity of Accuracy");

        for i in range(len(YF.columns)):

            print(str(YF.columns[i]) + ":\t"\
                    + str(delta_f[i]));

        print("L1 norm = " + str(sum(delta_f)));
        print("L2 norm = " + str(utilities.L2_norm(delta_f)));


        print("===========================================\n\n");

    return delta_f;

#######################################
def get_command():

    global COMMAND_STRS;
    #= ["exit", 
    #    "patient",
    #    "sensitivity",
    #    "machine",
    #    "test",
    #    "epsilon"];
    #   "write"
    #   "load"

    global COMMAND_INTS;

    global COMMAND_DESCS 
    #= ["0 - Exit",
    #    "1 - Compare Machine Result with Actual Patient",
    #    "2 - Perform Sensitivity Analysis on Current Machine",
    #    "3 - Select Another Machine",
    #    "4 - Run Accuracy Test",
    #    "5 - Set Epsilon on Copy of Current Machine"
    #   6 write machine
    #   7 load machine];

    #COMMAND_FUNCS = [];
    opt = -1;

    while opt not in COMMAND_INTS:

        print("\n===================================================");
        print();
        print("Analysis Command Menu");
        print();
        for i in range(len(COMMAND_DESCS) - 1):
            print(COMMAND_DESCS[i + 1]);
        print();
        print(COMMAND_DESCS[0]);
        print();
        print("===================================================");
        print();
        print();
        print("Your Selection from " + str(COMMAND_INTS) + ", please: ", end = "");
        sys.stdout.flush();

        try:
            opt = int(input());

        except:
            print("The response must be an integer, please.");

        if opt not in COMMAND_INTS:
            print(str(opt) + " is not a valid selection.");
            print("Please select an option from the menu.");

    return COMMAND_STRS[opt];

#######################################
def get_patient_command():

    result = "";

    while 0 == len(result):

        result = input("Please enter the four digit patient number to test "\
                "(or exit to quit): ")

        if 4 != len(result):

            print("Please enter a valid input...try again.");
            result = "";

        else:

            # suppose the user entered 'exit' or a four digit number (with 
            # possible leading zeros), if so pass otherwise reset result
            # and advise user to enter valid input

            if result.lower() == "exit":

                # done
                pass;

            else:

                for i in range(len(result)):

                    if result[i] not in string.digits:

                        print("Please enter a valid input...try again.");
                        result = "";
                        break;

    return result;

#######################################
def do_patient_command(pno, mchn, X, Y, patients):

    # test if there is a patient with pno; if so record it's index
    ndx = 0;

    while ndx < len(patients) and pno != patients[ndx].get_number():

        ndx += 1;

    # there was no match
    if ndx == len(patients):

        print("There was no patient data for that patient number. Please try again.");
        return;

    # print pixel data?
    # print(X.iloc[ndx:ndx + 1]);

    pred = mchn.predict(X.iloc[ndx:ndx + 1])[0];
    true, true_cols = patients[ndx].metadata2binary();

    #table, column_names = oas.patient_data.dataset_table(patients);
    column_names = true_cols;

    if DEBUG:
        print("pred: " + str(pred));
        print("true: " + str(true));
        print("column_names: " + str(column_names));

    # report predicted result
    print("\tPredicted Y\t\tActual Y (ASCII Encoded)");

    for i in range(len(column_names)):

        if i < len(pred) and i < len(true):

            print(column_names[i] + ":\t" + str(pred[i]) + "\t\t\t" + str(true[i]));

        else:
            break;

    # report actual patient data
    print("Actual Patient (Original Data)");
    print(patients[ndx].metadata);

    return;

#######################################
# name: name string of the machine to select
# titles: list of name string for available machines
# machines: list of predict/accuracy compatable objects
# Xs, Ys: list of X, Y dataframes for each machine
def set_machine(name, titles, machines, Xs, Ys, datasets, epsilons, entropys, types):

    ndx = titles.index(name);

    return titles[ndx],\
            machines[ndx],\
            Xs[ndx],\
            Ys[ndx],\
            datasets[ndx],\
            epsilons[ndx],\
            entropys[ndx],\
            types[ndx];

#######################################
# nm (list of strings): provide the names of the models
# mchn (list of ai models): the models
# Xs (list of X dataframes): model i's input values
# Ys (list of Y dataframes): model i's output values
# dataset (list of patients): original dataset patients
# types (list of integer codes): the type of intelligence used
def interaction_loop(path,\
        quantity,\
        proportion,\
        nm, mchn, Xs, Ys, dtsts, eps, subs, typs):
    # interaction loop:
    # * select patient (or exit)
    # * use corresponding x data to may a prediction y
    # * compare prediction y with actual y
    # * repeat

    name = "No Machine Loaded";

    machine = None;
    X = None;
    Y = None;
    dataset = None;
    epsilon = float();
    entropy = float();
    kind = None;

    accuracy = None;

    if 0 < len(nm):
        # default to the first machine in the set
        name, machine, X, Y, dataset, epsilon, entropy, kind =\
        set_machine(nm[0], nm, mchn, Xs, Ys, dtsts, eps, subs, typs);

    else:
        print("There are no machines to load yet....");

    # until user quits
    while True:

        # inform user of model being used
        print("\nUsing " + name + " Model");
        print("Entropy (Total): " + str(entropy));
        print("Epsilon: " + str(epsilon));
        print();

        # get a command from the user
        comm = get_command();

        #print("pnumber = " + str(pnumber));

        # did the user choose to exit?
        if "exit" == comm.lower():

            break;
        # additional options
        elif "patient" == comm.lower():

            pnumber = get_patient_command();

            if "exit" != pnumber.lower():

                do_patient_command(pnumber, machine, X, Y, dataset);

        elif "sensitivity" == comm.lower():

            # mchn, Xs, Ys, dtsts, typs
            if kind == DECISION_TREE:
                get_dt_bounded_privacy(\
                    mchn[0], Xs[0], Ys[0], Xs[1], Ys[1], dtsts[0], dtsts[1]);
                get_dt_unbounded_privacy(\
                    mchn[0], Xs[0], Ys[0], dtsts[0]);

            elif kind == NEURAL_NETWORK:
                print("Not implemented...");

            else:
                print("Unknown intelligence type.");

        elif "machine" == comm.lower():

            # prompt the user for the machine to use
            name = get_machine_command(nm);

            # update the current machine
            name, machine, X, Y, dataset, epsilon, entropy, kind =\
                set_machine(\
                    name,\
                    nm,\
                    mchn,\
                    Xs, Ys,
                    dtsts,\
                    eps,\
                    subs,\
                    typs);

        elif "test" == comm.lower():

            accuracy = test_model(machine, X, Y, kind);

        elif "epsilon" == comm.lower():

            # the collection of machines must have at least one entry
            if len(mchn) < 1:

                print("There must be at least one machine loaded.");
                print("Please load a machine....");

            # otherwise, replace the noisy machine at index 1
            else:

                name = "Noisy DT";
                print("Building new noisy machine/data from dataset located in: " + str(path));
                print("=== " + str(name) + " ===");
                e, s, m, x, y, d =\
                    build_decision_tree(path,\
                        quantity,\
                        proportion,\
                        noisy);

                # there is already a filled machine
                if len(mchn) > 1:
                    nm[1] = name;
                    mchn[1] = m;
                    Xs[1] = x;
                    Ys[1] = y;
                    dtsts[1] = d;
                    eps[1] = e;
                    subs[1] = s;

                else:
                    nm.append(name);
                    mchn.append(m);
                    Xs.append(x);
                    Ys.append(y);
                    dtsts.append(d);
                    eps.append(e);
                    subs.append(s);

        #elif "write" == comm.lower():

        #    utilities.write_machine(machine, X, Y, dataset, kind);

        #elif "load" == comm.lower():

        #    m, x, y, d, k = utilities.read_machine();

        #    new_name = "";

        #    while ("" == new_name or new_name in nm) and new_name.lower() != "quit":

        #        new_name = input("Please enter a distinct name for the new machine: ");

        #        if new_name in nm and new_name.lower != "quit":
        #            print("That name is already in use...");
        #            print("Please try again or type 'exit' to quit.");

            # update: name, machine, X, Y, dataset, kind
        #    if new_name.lower() != "quit":

        #        name = new_name;
        #        nm.append(new_name);

        #        machine = m;
        #        mchn.append(m);

        #        X = x;
        #        Xs.append(x);

        #        Y = y;
        #        Ys.append(y);

        #        dataset = d;
        #        dtsts.append(d);

        #        kind = k;
        #        typs.append(k);
        else:

            print("Command: " + comm + " is not implemented.");

    return;

#######################################
def usage(exe):

    print("usage: " + exe + " -d <path to data>");

    return;

#######################################
def main(argc, argv):

    # default size and proportion for this main program
    sz = 500;
    fraction = 0.8;
    epsilon = float();
    entropy = float();
    path = None;

    # take care that there is a compatible
    # machine available to load
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
    epsilons = list();
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

            # ************************************
            # not used
            elif op in ("-e", "--epsilon"):

                try:
                    epsilon = float(ar);

                except:
                    print("Epsilon must be a floating point number.");
                    return 3;
            # ************************************

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
                epsilon, entropy, mach, X, Y, P =\
                    build_decision_tree(path, sz, fraction, filled);
                ai_type = DECISION_TREE;    
                add = True;

            elif "Noisy Decision Tree" == names[i]:

                # run the build to create the noisy dataset
                epsilon, entropy, mach, X, Y, P =\
                    build_decision_tree(path, sz, fraction, noisy);
                ai_type = DECISION_TREE;
                add = True;

            else:

                # report that the machine build is not ready
                print("The machine: " + name[i]\
                + "'s build routine is not ready.");

            if True == add:

                # update machine attributes
                machines.append(mach);
                Xs.append(X);
                Ys.append(Y);
                datasets.append(P);
                epsilons.append(epsilon);
                entropys.append(entropy);
                types.append(ai_type);

    # skip building models
    else:
        print("Loading models from ./ (current working directory) as requested...");

    print("===========================================\n\n");

    # provide the names of the models, the model, the Xs
    # and Ys for the model in parallel lists, and 
    # the dataset of patients
    interaction_loop(\
            path,\
            sz,\
            fraction,\
            names,\
            machines,\
            Xs, Ys,\
            datasets,\
            epsilons,\
            entropys,\
            types);

    return 0;

if "__main__" == __name__:

    result = main(len(sys.argv), sys.argv);
    exit(result);


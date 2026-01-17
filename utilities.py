import shutil;
import random;
import sys;
import pathlib;
import os;
import time;
import pickle;
import numpy as np;
import math;

# file extensions so that original + additional is the new filename
# original file extensions
ORIGINAL_META_EXT = "MR1.txt";

# new file extensions
ADDITIONAL_META_EXT = ".noise.txt";

# fill file extensions
FILL_META_EXT = ".fill.txt"

# turn on debugging
DEBUG = False;

###################################################
# remove leading zeros from the string parameter
def unpad(val):
    
    result = str();
    start = True;
    keep = False;
    
    for i in range(len(val)):
        
        # is it the last zero
        if i == len(val) - 1:
            
            # take the place holder
            result += val[i];
            
        # is it a leading zero?
        elif True == start and '0' == val[i]:
            
            # go to the next char
            pass;
            
        elif '0' != val[i]:
            
            start = False;
            result += val[i];
            
    return result;
            
###################################################
# return L2 norm of the given list
def L2_norm(mylist):

    # intermediate result
    re = float();

    # sum the squares
    for e in mylist:
        re += e**2.0;

    # take the square root
    return math.sqrt(re);

###################################################
# returns maximum delta(f) aka sensitivity
# for each column in x and y which must be numeric data.
#
# uses range for aggregate function
def get_range_per_column(X, Y):

    X_range = list();
    Y_range = list();

    maximum = int();
    minimum = int();

    # get range on the X values, call it sensitivity
    for col in X.columns:

        maximum = X[col].max();
        minimum = X[col].min();

        X_range.append(maximum - minimum);

    # get range on the Y values, call it sensitivity
    for col in Y.columns:

        maximum = Y[col].max();
        minimum = Y[col].min();

        Y_range.append(maximum - minimum);

    return X_range, Y_range;

###############################################
# Laplace distribution with zero mean and 
# scale equal to sensitivity / epsilon
def laplace_noise(value, sensitivity, epsilon):
    scale = sensitivity / epsilon;
    noise = np.random.laplace(0, scale, 1)[0];
    return value + noise;

###############################################
# save an object to file
def write_to_disk(file, obj):

    file.write(pickle.dumps(obj));

    return;

###############################################
def load_from_file(file):

    blob = file.read();
    obj = pickle.loads(blob);

    return obj;

###############################################
def write_machine(dt, x, y, dset, t):

    filename = input("Please enter a filename "\
            + "(it will be overwritten): ");

    # the tree/network
    try:
        objfile = open("Machine" + filename, "wb");

    except:
        print("File could not be written.");
        return;
    write_to_disk(objfile, dt);
    objfile.close();

    # Xs
    try:
        objfile = open("X" + filename, "wb");

    except:
        print("File could not be written.");
        return;
    write_to_disk(objfile, x);
    objfile.close();

    # Ys
    try:
        objfile = open("Y" + filename, "wb");

    except:
        print("File could not be written.");
        return;
    write_to_disk(objfile, y);
    objfile.close();

    # dataset
    try:
        objfile = open("Dataset" + filename, "wb");

    except:
        print("File could not be written.");
        return;
    write_to_disk(objfile, dset);
    objfile.close();

    # type
    try:
        objfile = open("Type" + filename, "wb");

    except:
        print("File could not be written.");
        return;
    write_to_disk(objfile, t);
    objfile.close();

    return;

###############################################
def read_machine():

    machine = None;
    x = None;
    y = None;
    dataset = None;
    kind = None;

    objfile = None;

    filename = input("Please enter a filename "\
            + "from which to read: ");

    # machine
    try:
        objfile = open("Machine" + filename, "rb");

    except:
        print("File could not be read.");
        return machine, x, y, dataset, kind;
    machine = load_from_file(objfile);
    objfile.close();

    # x
    try:
        objfile = open("X" + filename, "rb");

    except:
        print("File could not be read.");
        return machine, x, y, dataset, kind;
    x = load_from_file(objfile);
    objfile.close();


    # y
    try:
        objfile = open("Y" + filename, "rb");

    except:
        print("File could not be read.");
        return machine, x, y, dataset, kind;
    y = load_from_file(objfile);
    objfile.close();

    # dataset
    try:
        objfile = open("Dataset" + filename, "rb");

    except:
        print("File could not be read.");
        return machine, x, y, dataset, kind;
    dataset = load_from_file(objfile);
    objfile.close();

    # type
    try:
        objfile = open("Type" + filename, "rb");

    except:
        print("File could not be read.");
        return machine, x, y, dataset, kind;
    kind = load_from_file(objfile);
    objfile.close();

    return machine, x, y, dataset, kind;

###############################################
# produce a unique random number given a range
# a list of randoms that have already been used
def unique_random(low, high, used):
    
    ndx = random.randint(low, high);

    while ndx in used:

        ndx = random.randint(low, high);

    return ndx;

###############################################
# test for end of file
def is_eof(file):

    curr = file.tell();
    file.seek(0, os.SEEK_END);
    end = file.tell();
    file.seek(curr, os.SEEK_SET);

    return curr == end;

###############################################
# extract patient number from filename, if possible
def getn(name):

    # reset the location and the number
    location = -1;
    number = str();

    # attempt to locate the first underscore in the name of the file
    try:
        location = str(name).index("_");

    # if there is no underscore there is a problem
    except ValueError:
        pass;

    if location != -1:

        # extract the patient number from the filename
        number = name[location + 1:location + 5];
        
    return number;

################################################
# write to file with filename (including path) that includes stuff_to_write
def insert_beginning(filename, stuff_to_write):

    if len(stuff_to_write) == 0:

        print("The content being inserted into " + filename + " must be nonempty.");
        return 1;

    # the file to update
    myfile = open(filename, "w");

    myfile.write(stuff_to_write);

    #close the file
    myfile.close();

    return 0;

################################################
def match_list(strng, lst):

    if len(strng) != len(lst):

        return False;

    for i in range(len(strng)):

        if strng[i] != lst[i]:

            return False;

    return True;

################################################
def shift_list_left(lst):

    for i in range(len(lst) - 1):

        lst[i] = lst[i + 1];

    return lst;

################################################
# write to file with filename (including path) that includes stuff_to_write
def insert_beginning_append(oldfilename, newfilename, stuff_to_write):

    if len(stuff_to_write) == 0:

        print("The content being inserted into " + filename + " must be nonempty.");
        return 1;

    if len(oldfilename) == 0:

        print("The source file must be specified.");
        return 2;

    if len(newfilename) == 0:

        print("The target file must be specified.");
        return 3;

    # the file to update
    myoldfile = open(oldfilename, "r+");

    # move to beginning of the file
    myoldfile.seek(0, 0);

    # record the first line of the file (session id)
    session = str();

    while True:
        
        # read one character
        content = myoldfile.read(1);

        # is it the new line or did it reach the end of file
        if '\n' == content or is_eof(myoldfile):

            break;

        # otherwise add the character to the session information
        # move to the next
        session += content;

    # keep the session id on its own line
    session += "\n"

    if is_eof(myoldfile):

        # there was a problem with the file format
        print("There was a problem reading the session id from "\
        + oldfilename + " for fill function.");
        return 5;


    # locate first occurance 'mpr-1'
    # aka the sliding window problem; create a list as long as the string
    # to match
    myoldfile.seek(0,0)

    mpr_match = ['0', '0', '0', '0', '0'];

    # create the search key
    mpr_key = ['m', 'p', 'r', '-', '1'];

    while not is_eof(myoldfile):

        # shift list content left
        shift_list_left(mpr_match);

        # update last entry
        mpr_match[len(mpr_match) - 1] = myoldfile.read(1);

        # test for a match
        if match_list(mpr_key, mpr_match):

            # rewind the file read location to the location of mpr-1
            break;

    if is_eof(myoldfile):

        print("There was a problem reading the mpr-1 marker from "\
        + oldfilename + " for fill function.");
        return 6;

    # create fill file
    mynewfile = open(newfilename, "w");

    mynewfile.write(session);

    mynewfile.write(stuff_to_write);

    mynewfile.write("\n\nmpr-1");

    while not is_eof(myoldfile):

        character = myoldfile.read(1);
        mynewfile.write(character);
        
    #close the files
    myoldfile.close();
    mynewfile.close();

    return 0;

###########################################
# get the list of metafiles from subdirectories in the given path
def get_metafile_list(path, ext):
    # get the list of files in the given path
    allfiles = traverse(path);

    # filter the out the original metadata files
    metafiles = list();

    for file in allfiles:
    
        if ext == file[(-1) * len(ext):]:

            metafiles.append(file);

    if DEBUG:
        # debug print list of original metafiles
        print_list(metafiles);

    return metafiles;

###########################################
# create a new text file with the noise from one patient
def add_overwrite(path, ext, patients):

    print("Processing " + path + " for " + ext + " files.");

    metafiles = get_metafile_list(path, ORIGINAL_META_EXT);

    # copy the files to include the additional extension
    for file in metafiles:

        number = getn(file);

        for p in patients:

            if p.number == number:

                insert_beginning_append(file, file + ext, str(p.metadata));

    return;

###########################################
# create a new text file with the noise from one patient
def add_once(path, ext, patients):

    print("Processing " + path + " for " + ext + " files.");

    metafiles = get_metafile_list(path, ORIGINAL_META_EXT);

    for file in metafiles:

        number = getn(file);

        for p in patients:

            if p.number == number and not os.path.exists(file + ext):

                insert_beginning_append(file, file + ext, str(p.metadata));

    return;

#######################################
# test that all of the table rows are the same length
def test_trim(table):

    row_len = [len(r) for r in table];

    return min(row_len) == max(row_len);

#######################################
# add/remove columns (valued 0x00), from the end of the table's 
# rows, until each row is DATASET_COLUMNS in length
# example: table = utilities.trim_2d_columns(table, DATASET_COLUMNS);
def trim_2d_columns_zeros(matrix, row_len):

    for i in range(len(matrix)):

        j = len(matrix[i]);

        while j > row_len:

            del matrix[i][j - 1];
            j -= 1;

        while j < row_len:

            matrix[i].append(0x00);
            j += 1;
         
    return matrix;

#######################################
# add/remove entries (nil0, nil1, etc.) from a list
# until it is DATASET_COLUMNS in length
# example: titles = utilities.trim_list(titles, DATASET_COLUMNS);
def trim_list_nils(arr, row_len):

    pad = "px";

    j = len(arr);

    while j > row_len:

        del arr[j - 1];
        j -= 1;

    while j < row_len:

        arr.append(pad + str(j));
        j += 1;
         
    return arr;

####################################
# print a list
def print_list(lst):

    for e in lst:

        print(e)

    return

#####################################
# return a list of files in the given directory and each nonempty subdirectory
def traverse(d):

    result = list();

    p = pathlib.Path(d);

    tree = [x for x in p.iterdir()];

    tree.sort();

    for x in tree:

        if x.is_dir():

            result += traverse(x);

    for x in tree:

        if not x.is_dir():

            result.append(str(x));

    return result;

#######################################
# print a matrix with a title and labels for the rows and columns
def print_square_matrix_with_titles(mat, label, title):

    print("Matrix size:" + str(mat.shape));

    for nm in label:

        print(str(nm) + "\t", end = "");

    print();

    for i in range(mat.shape[0]):

        for j in range(mat.shape[1]):

            #print("i: " + str(i) + " j: " + str(j));
            print(str(round(mat.iat[i, j], 2)) + "\t", end = "");

        print()

#######################################
# print a matrix with a title and labels for the rows and columns
def write_square_matrix_with_titles(mat, label, title, filename):

    myfile = open(filename, "w");

    myfile.write("Matrix size:" + str(mat.shape) + "\n");

    for nm in label:

       myfile. write(str(nm) + "\t");

    myfile.write("\n");

    for i in range(mat.shape[0]):

        for j in range(mat.shape[1]):

            myfile.write(str(round(mat.iat[i, j], 2)) + "\t");

        myfile.write("\n");

    myfile.close();

    return;

#######################################
# get the current time for distinct filenames
def get_timestamp():

    return time.time_ns();

#######################################
# add noise files reflecting the random values from the dataset
def add_noise_files(path, dataset):

    add_overwrite(path, ADDITIONAL_META_EXT, dataset);

    if DEBUG:
        for patient in dataset:

            print(patient.metadata);
            print("=======================\n");

    return;

#######################################
# add fill files reflecting the random values from the dataset
# and the remaining original data
def add_fill_files(path, dataset):

    add_once(path, FILL_META_EXT, dataset);

    return;


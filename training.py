#!/usr/bin/python3

import sys;
import numpy as np;
import random;
import os;
import copy;
import string;

import patient;
import flatten;
import utilities as uti;

IMAGE_EXT = "mpr-1_anon_sag_66.gif";
META_EXT = uti.ORIGINAL_META_EXT; # + uti.FILL_META_EXT; #uti.ADDITIONAL_META_EXT;

DEBUG = False;

class session_data:

    # dimension and dimension_possible_values must have the same
    # top level number of dimensions

    dimension = [
        "AGE:",     # 0
        "M/F:",     # 1
        "HAND:",    # 2
        "EDUC:",    # 3
        "SES:",     # 4
        "CDR:",     # 5
        "MMSE:",    # 6
        "eTIV:",    # 7 -- begins anatomically bias measurements
        "ASF:",     # 8
        "nWBV:" ];  # 9

    first_floating_estimate = 7; # mark anatomically bias measurements

    # dimensions that come from a known set of values, i.e. discrete quantities
    dimension_possible_values_general = [
            ["{:06d}".format(x) for x in range(10, 100)],    # age generally available and can be randomly assigned
            ["00Male", "Female"],             # gender
            ["0Left", "Right"],              # handedness
            ["1", "2", "3", "4", "5"],      # education level
            ["1", "2", "3", "4", "5"],      # socieconomic situation (SES); none were given in the guide
            ["000", "0.5", "001", "002"],         # clinical dementia rating (CDR)
            ["{:02d}".format(x) for x in range(31)] ];  # mini-mental state examination (MMSE)

    # the male values for mean and variance for a given category are given first
    # then the corresponding mean and variance for a given category are given
    # a female experimental unit
    # the values are estimated from:
    # https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2020.00278/full
    # figure 1
    #
    # see also:
    # https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/1471-2288-5-13
    # regarding the calculation of variance from the range information
    dimension_possible_values_bias = [
        [ 1350, ((1.60 - 1.10) / 4.0) * 1000 ],     # eTIV mean 1.35 L 
                                                    # (1350 mL) and range 
                                                    # 1.10 - 1.60 liters 
                                                    # (1100 - 1600 mL)
        [ 1500, ((1.88 - 1.25) / 4.0) * 1000 ],     # female
        [ (1.56 + 0.88) / 2.0, ((1.56 - 0.88) / 4.0) ], # male and female atlas
                                                        # scale factor
        [ (1.56 + 0.88) / 2.0, ((1.56 - 0.88) / 4.0) ],
        [ 1.100, ((1.25 - 0.8) / 4) ],        # nWBV (liters) male
        [ 1.260, ((1.51 - 1.0) / 4) ] ];      # female
        
    # Define intervals for each attribute (example intervals)
    intervals_dict = {
        'AGE:': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'EDUC:': [1, 2, 3, 4, 5],
        'SES:': [1, 2, 3, 4, 5],
        'CDR:': [0, 0.5, 1, 2],
        'MMSE:': [0, 5, 10, 15, 20, 25, 30],
        'eTIV:': [800, 1000, 1200, 1400, 1600, 1800, 2000],
        'ASF:': [0.7, 0.9, 1.1, 1.3, 1.5, 1.7],
        'nWBV:': [0.7, 0.9, 1.1, 1.3, 1.5, 1.7]
        
    }

    # substitution counts by category
    dimension_substitution = [0 for d in dimension];

    # obfuscation counts by category
    dimension_obfuscation = [0 for d in dimension];

    def __init__(self):

        self.attr = session_data.dimension.copy();

    def __str__(self):

        result = str()

        for i in range(len(session_data.dimension)):
            
            tmp_attr = uti.unpad(self.attr[i]);

            result += session_data.dimension[i] + "\t" + str(tmp_attr); # self.attr[i]

            if len(session_data.dimension) - 1 != i:

                result += "\n"

        return result

    # fill data at ndx in this patient record
    # using either the Laplace Mechanism (noise) for the noisy set: 
    # F(x) = f(x) + lap(sensitivity / epsilon)
    # or if sensitivity = epsilon = -1, this function creates filled
    # data for blank attributes and resulting in the baseline dataset
    def fill_dimension(self, ndx, sensitivity = dict(), epsilon = -1.0):

        if sensitivity == dict() and epsilon == -1.0:
            # if this category calls for a value from a fixed set
            if ndx < session_data.first_floating_estimate:

                # select a value from the middle of the data's possible values
                sel = len(session_data.dimension_possible_values_general[ndx]);
                
                # if the fixed set is defined in the table
                # start with the median
                if sel > 0:
                    
                    if sel % 2 == 0:
                        
                        median = (float(session_data.dimension_possible_values_general[ndx][sel//2])\
                            + float(session_data.dimension_possible_values_general[ndx][sel//2 - 1])) / 2.0
                            
                    else:
                        
                        median = float(session_data.dimension_possible_values_general[ndx][sel//2])
                        
                    # select one of the values randomly
                    #self.attr[ndx] = str(random.choice(\
                    #session_data.dimension_possible_values_general[ndx])).zfill(8);
                    
                    new_value = int(np.random.laplace(\
                    loc = median,\
                    scale = (float(session_data.dimension_possible_values_general[ndx][-1])\
                        - float(session_data.dimension_possible_values_general[ndx][0])) / 1.0,\
                    size = 1)[0]);
                    
                    # clipping needs to happen here
                    neigh = 0
                    
                    while neigh < sel and\
                        new_value < float(session_data.dimension_possible_values_general[ndx][neigh]):
                            
                        neigh += 1;
                        
                    if neigh == 0:
                        
                        new_value = float(session_data.dimension_possible_values_general[ndx][0])
                        
                    elif neigh == sel:
                        
                        new_value = float(session_data.dimension_possible_values_general[ndx][-1])

                    else: # new value is with discrete ,qulatiative classes

                        low= new_value-float(session_data.dimension_possible_values_general[ndx][neigh-1])
                        high= float(session_data.dimension_possible_values_general[ndx][neigh])- new_value

                        if low < high:
                            new_value= float(session_data.dimension_possible_values_general[ndx][neigh-1])
                        else: # low is larger than or equal to high
                            new_value= float(session_data.dimension_possible_values_general[ndx][neigh])
                        
                    self.attr[ndx] = "{:06d}".format(int(new_value));


                # the value from a fixed set is not defined and generated randomly
                else:

                    num = int(abs(np.random.laplace(\
                    loc = (1000 + 9999) / 2,\
                    scale = (9999 - 1000) / 4,\
                    size = 1)[0]));

                    while num < 1000 or num > 9999:

                        num = int(abs(np.random.laplace(\
                        loc = (1000 + 9999) / 2,\
                        scale = (9999 - 1000) / 4,\
                        size = 1)[0]));

                    self.attr[ndx] = "{:06d}".format(num);

            # otherwise this value is generated using a gender based
            # random value assuming a Laplace distribution
            else:

                # hope that the gender is already specified (as it is 
                # listed in the dimensions prior to the gender biased 
                # values)

                # locate the gender pair in the bias dimension
                index = ndx - session_data.first_floating_estimate;

                # corrected data coupling with index function
                gender_ndx = session_data.dimension.index("M/F:");

                # test if the subject is male or not
                if self.attr[gender_ndx] == "Male".zfill(6):

                    self.attr[ndx] = "{:09.2f}".format(abs(np.random.laplace(\
                    loc = session_data.dimension_possible_values_bias[2 * index][0],\
                    scale = session_data.dimension_possible_values_bias[2 * index][1],\
                    size = 1)[0]));

                # female
                else:

                    self.attr[ndx] = "{:09.2f}".format(abs(np.random.laplace(\
                    loc = session_data.dimension_possible_values_bias[2 * index + 1][0],\
                    scale = session_data.dimension_possible_values_bias[2 * index + 1][1],\
                    size = 1)[0]));

        # sensitivity != -1 or epsilon != -1 ===>
        # do a fuzz on the data that already exists according to the
        # Laplace Mechanism (noise) from Programming Differential Privacy:
        # F(x) = f(x) + lap(s/e)
        else:

            verified = False;

            if DEBUG:

                if not verified:
                    verified = True;
                    print("Got Laplace Mechanism (noise) "\
                            + "from Programming Differential Privacy.");
            aname = session_data.dimension[ndx];
            vscale = sensitivity[aname][1] - sensitivity[aname][0]
            new_value = float(self.attr[ndx]) + abs(np.random.laplace(\
                    loc = 0, # the mean, mu
                    scale = vscale / epsilon, # the scale: lambda
                    size = 1)[0]); # the number of elements and which to select

            # if this category pads to six digits
            if ndx < session_data.first_floating_estimate:

                try:
                    dim_nos = [float(x) for x in session_data.dimension_possible_values_general[ndx]];
                    
                except ValueError:
                    dim_nos = [0.0];
                    
                dim_max = max(dim_nos);
                dim_min = min(dim_nos);

                # clip the random values back into valid range
                if new_value > dim_max:
                    
                    new_value = dim_max;
                    
                if new_value < dim_min:
                    
                    new_value = dim_min;
                
                self.attr[ndx] = "{:06d}".format(int(new_value));
                
            # otherwise it pads to nine digits
            else:

                self.attr[ndx] = "{:09.2f}".format(new_value);
        return;

    # update the relevant dimension, if applicable from the line of input
    def search_line_for_dimension(self, line):

        # for every recorded dimension
        for i in range(len(session_data.dimension)):

            # if it is mentioned on the current line
            if session_data.dimension[i] in line:

                # update it if it differs from an empty string,
                # otherwise keep the name of the dimension
                values = line.split()

                # if the metadata included a measurement,
                # update the attribute. otherwise, the attribute's
                # value remains equal to the dimension's name
                if len(values) > 1:

                    # one of the qualitative measurements
                    if i < session_data.first_floating_estimate:

                        self.attr[i] = str(\
                                values[1]).zfill(6);

                    # one of the quantitative measurements
                    else:
                        
                        self.attr[i] = str(\
                                values[1]).zfill(9);

                # create a random value for the category that was left blank
                else:
                    
                    # update the count
                    session_data.dimension_substitution[i] += 1;

                    # fill the dimension data
                    self.fill_dimension(i);

        return;

    # ===========================================================
    #######################################
    def print_2column(title, counter):

        print(title + ":");
        print("Category\t" + title);
    
        for i in range(len(session_data.dimension)):

            print(session_data.dimension[i] + "\t\t" + str(counter[i]));

        print("Total:\t\t" + str(sum(counter)));

        return;

    def reset_update_counters():

        # substitution counts by category
        dimension_substitution = [0 for d in session_data.dimension];

        # obfuscation counts by category
        dimension_obfuscation = [0 for d in session_data.dimension];

class patient_data:

    # ============================================================
    def __init__(self):

        # patient number
        self.number = str()

        # path to text file name
        self.metadata = session_data()

        # list of pathes to image files
        self.image_file = list()

        # content of image files as byte arrays
        self.image = list()

        return;

    # ============================================================
    def __str__(self):

        sample = 200

        result = "======================================\n"

        # patient number
        result += "Patient: " + self.number + "\n"

        # metadata keys
        result += "\nMetadata:\n" + str(self.metadata) + "\n"

        result += "\nBinary Metadata:\n"

        tmp_meta, tmp_cols = self.metadata2binary()

        for i in range(len(tmp_meta)):

            result += format(tmp_meta[i], "02x") + " "

            if 0 == (i + 1) % 20:

                result += "\n"

        result += "\n"

        for i in range(len(tmp_meta)):

            result += format(tmp_meta[i], "02d") + " ";

            if 0 == (i + 1) % 20:

                result += "\n";

        result += "\n";

        # image file data
        for i in range(len(self.image_file)):

            # file name
            result += "\nImage " + str(i + 1) + ": " + self.image_file[i] + "\n"
            result += "Bytes: " + str(len(self.image[i]) // 2 - sample // 2) + \
            " thru " + str(len(self.image[i]) // 2 - sample // 2 + sample) + "\n" 

            # first 200 bytes
            
            for j in range(sample):

                result += format(self.image[i][j + (len(self.image[i]) // 2 - sample // 2)], "02x") + " "

                if 0 == (j + 1) % 20:

                    result += "\n"

            result += "\n"
        

        result += "======================================\n"

        return result;

    # ==============================================================
    # set num to integer n
    def set_number(self, n):

        self.number = patient_data.int2patient(n)
        return

    # ==============================================================
    # get string version of patient number
    def get_number(self):

        return self.number

    # ===============================================================
    # get the text file path, for now
    def set_metadata(self, txt_file_path):

        file_data = open(txt_file_path, "r")

        for line in file_data:

            self.metadata.search_line_for_dimension(line)

        file_data.close()

        return

    # ===============================================================
    # get the collection of image file pathes
    def add_image(self, img_file_path):

        # extract binary data from the gif file
        flat_image = flatten.flatten_bin(img_file_path);

        # add the binary data to the list of binaries
        self.image.append(flat_image);

        # add the file name corresponding to the binary data in order
        self.image_file.append(img_file_path);

        # can not sort because it will lose relationship with binary data
        # does it matter if the filename position matches it's binary data
        # self.image_file.sort()

        return

    # ================================================================
    def metadata2binary(self):

        if DEBUG:
            # debug
            print("\npatient_data.metadata2binary");

        result = list();
        column = list();

        # for every attribute string
        for i in range(len(self.metadata.attr)):

            try:
                my_attr = bytearray(self.metadata.attr[i], "UTF-8");

            except:
                print("failed to get bytearray from self.metadata.attr["\
                        + str(i) + "]");

            # for the length of the attribute string
            for j in range(len(my_attr)):

                column.append(session_data.dimension[i] + str(j));
                result.append(my_attr[j]);

        # update columns
        if DEBUG:
            print("len(column) = " + str(len(column)));
                    

        if DEBUG: 
            # debug
            print("len(result) = " + str(len(result)));

        return result, column;

    # ======================================================================
    # image data to binary
    def image2binary(self):

        if DEBUG: 
            # debug
            print("\npatient_data.image2binary");
            print("len(self.image) = " + str(len(self.image)));

            if len(self.image) > 0:

                for i in range(len(self.image)):

                    print("len(self.image[" + str(i) + "]) = " + str(len(self.image[i])));

        result = list();
        column = list();
        total = int();

        # for every image
        for i in range(len(self.image)):

            my_im = bytearray(self.image[i]);

            # for the length of the current image
            # construct a column starting at px0
            for j in range(len(my_im)):

                column.append("px" + str(total));
                total += 1;
                
                result.append(my_im[j]);

        if DEBUG: 
            # debug
            print("len(column) = " + str(len(column)));
        
        if DEBUG: 
            # debug
            print("len(result) = " + str(len(result)));

        return result, column;

    # ======================================================================
    # take a list of binary encoded characters and turn it into a string
    def binary2string(lst):

        result = str();

        for d in lst:

            result += "{:2x}".format(d);

        return result;

    # =======================================================================
    # good up to 4 digits
    # returns zero for negative numbers and those with more than four digits
    def int2patient(n):

        # take a string representation of the number n
        result = str(n);

        # is it a negative number
        if 0 > n:

            result = "0000"

        # is it a single, positive digit
        elif 10 > n:

            result = "000" + result

        # is it two, positive digits
        elif 100 > n:

            result = "00" + result

        # is it three, positive digits
        elif 1000 > n:

            result = "0" + result

        # four digits are done, but if it more than that there is a problem
        elif 9999 < n:

            result = "0000"

        return result
    
    # =========================================================================
    # takes a list of patient_data objects, extracts the X values (metadata)
    def patient_metavalues(patient_set):

        result = list()
        metadata = list()

        row = list();
        column_names = list();
        maximal_column_names = list();

        # get every patients metadata, as an array of byte valued integers
        for p in patient_set:
            
            row, column_names = p.metadata2binary();

            if len(column_names) > len(maximal_column_names):

                maximal_column_names = column_names;

            metadata.append(row);

        len_metas = [len(r) for r in metadata];
        max_len_metas = max(len_metas);

        # ensure uniform lengths for each row
        for i in range(len(metadata)):

            while len_metas[i] < max_len_metas:

                metadata[i].append(0x00);
                len_metas[i] += 1;

        result = np.array(metadata);

        return result, maximal_column_names;

    # =====================================================================
    # takes a list of patient_data objects, extracts the Y values (image)
    def patient_imgvalues(patient_set):

        result = list();
        image_data = list();

        row = list();
        column_names = list();
        maximal_column_names = list();

        # for each patient in the set
        # side effect: updates MAX_PIXEL_COLUMNS
        for p in patient_set:

            row, column_names = p.image2binary();

            if len(column_names) > len(maximal_column_names):

                maximal_column_names = column_names;

            image_data.append(row);

        len_ids = [len(r) for r in image_data];
        max_len_ids = max(len_ids);

        # for each image, make length homogeneous
        for i in range(len(len_ids)):
        
            while len_ids[i] < max_len_ids:

                image_data[i].append(0x00);
                len_ids[i] += 1;

        # convert to a 2d numpy array
        result = np.array(image_data);

        return result, maximal_column_names;

    # ==================================================================
    # homogenize image data

    # ==================================================================
    # return the flat dataset table and the column names
    def dataset_table(patient_set, size = -1):

        if DEBUG: 
            # debug
            print("\ndataset_table");

        # construct the tabular data from the x, y data
        table = list();
        cols = list();
        
        # data table
        xdata, xcol = patient_data.patient_imgvalues(patient_set);
        
        if DEBUG: 
            # debug
            print("len(xdata) = " + str(len(xdata)));
            print("len(xcol) = " + str(len(xcol)));
        
        ydata, ycol = patient_data.patient_metavalues(patient_set);

        if DEBUG: 
            # debug
            print("len(ydata) = " + str(len(ydata)));
            print("len(ycol) = " + str(len(ycol)));

        cols = ycol + xcol;

        # for every row
        for i in range(len(ydata)):
            
            row = list();

            for r in ydata[i]:
                row.append(r);

            for r in xdata[i]:
                row.append(r);

            table.append(copy.deepcopy(row));

        if DEBUG:
            # debug
            print("Number of table rows: " + str(len(table)));
            print("Number of table value column names: " + str(len(cols)));

        return table, cols;

# ================================================================
# build training dataset using first sz elements of the dataset
def training_dataset(directory, sz):

    patients = list();

    percent = int();
    ppercent = int();

    if 0 == sz:

        return patients;

    for i in range(sz):

        num = patient_data.int2patient(i + 1);

        # debug: verify the patient numbers
        #print("patient num: " + str(num));

        images = patient.get_patientn(directory, num, IMAGE_EXT);
        text = patient.get_patientn(directory, num, META_EXT);

        if len(images) > 0 and len(text) > 0:

            p = patient_data();

            p.set_number(i + 1);

            if os.path.exists(text[0] + uti.FILL_META_EXT):

                p.set_metadata(text[0] + uti.FILL_META_EXT);

            else:

                p.set_metadata(text[0]);

            for x in images:
            
                p.add_image(x);

            patients.append(p);

        percent = int( ((i + 1) / sz) * 100.0 );

        if percent - ppercent >= 10:

            ppercent = percent;
            print(str(percent) + "%", end = "");

            if percent != 100:

                print(" ...", end = "");

            else:

                print();

            sys.stdout.flush();

    return patients;

# ===========================================================
# obfuscate exactly one patient
# dataset: list of patient datatype
# ndx: integer index within the list
# epsilon: the user specified epsilon for varying user data
def obfuscate_one(sensitivities, epsilon, dataset, ndx):

    # create list of index for dimensions to obfuscate
    dims = list();

    # add the age, gender and brain volume in that order
    # qualitative data elements
    dims.append(session_data.dimension.index("AGE:"));
    dims.append(session_data.dimension.index("EDUC:"));
    dims.append(session_data.dimension.index("SES:"));
    dims.append(session_data.dimension.index("CDR:"));
    dims.append(session_data.dimension.index("MMSE:"));
    # quantitative data elements
    dims.append(session_data.dimension.index("eTIV:"));
    dims.append(session_data.dimension.index("ASF:"));
    dims.append(session_data.dimension.index("nWBV:"));

    if DEBUG:
        # debug print dims
        print("utilities.obfuscate: The indice in dims to obscure: ");
        uti.print_list(dims);
    
    # select how many of the demographic categories
    # to update...
    # changed to: always all dimensions in list dims
    obfuscate = len(dims); # random.randint(1, len(dims));

    # for the number of attributes to obfuscate
    for j in range(obfuscate): # range(obfuscate): -- decided to change all

        # update demographic value
        dataset[ndx].metadata.fill_dimension(\
                dims[j],\
                sensitivities,\
                epsilon);

        # update the obfuscation count
        session_data.dimension_obfuscation[dims[j]] += 1;

    # return the list of patients with the newly obfuscated element
    return dataset;

# ===========================================================
# obfuscate random patients in the given dataset
# to the epsilon level
def obfuscate(fraction, epsilon, dataset):

    # copy the original dataset
    myds = copy.deepcopy(dataset);

    # find a 'nice' (integer) percentage of the dataset
    # always 100% for this application
    num_to_obscure = int(len(myds) * fraction);

    # determine each dimension's sensitivity
    sensitivities = dict();

    # intermediate value to hold numeric data
    # which will be obscured
    val = float();

    # boundary values
    minimum = float();
    maximum = float();

    # flag for string data exception
    change = bool();

    # find the range on the data actually read
    # from the file i.e.
    # for each recorded dimension
    for ndx in range(len(session_data.dimension)):

        # min is initially large and max is initially small
        minimum = 9999999; # initially a very large number
        maximum = float(); # initially a very small number

        # check if any patient updates the min and/or max
        for p in myds:

            # suppose it should be obscured
            change = True;

            # attempt to convert the data to a numeric type
            try:
                val = float(p.metadata.attr[ndx]);

            except:
                # the data contained alphabetical characters
                # so keep it...
                change = False;

            # if it is numeric data, then update min/max
            if change:

                if val < minimum:
                    minimum = val;

                if val > maximum:
                    maximum = val;

            # otherwise string data does not have a range, yet...
            else:
                maximum = minimum = 0.0;

        # record the range of the values;
        # strings always produce 0.0
        sensitivities[session_data.dimension[ndx]] = (minimum, maximum);

    # for the number of patients to obscure
    # i.e. 100% to create a noisy data set
    for i in range(num_to_obscure):

        # obfuscate exactly that one patient
        myds = obfuscate_one(sensitivities, epsilon, myds, i);

    return myds;

# ===========================================================
def main(argc, argv):

    sz = 3

    # debug: get argument count
    print("argument count: " + str(argc));

    # debug: get argument list
    print("argument list: ");
    uti.print_list(argv);

    if argc < 2:
        print("usage: " + sys.argv[0] + " <path to oasis data> [set size (default = " + str(sz) + ")]")
        exit(1)

    if 3 == argc:

        try:
            sz = int(argv[2])

        except:
            pass

    # debug: get the size requested by user
    print("size requested by user: " + str(sz));

    # extract training data from the disc1 directory for
    # first 3 patients
    patients = training_dataset(argv[1], sz)

    for p in patients:

        print(p)

    # convert training data into a flat table
    columns = patient_data.dataset_table_column_names(patients);
    table = patient_data.dataset_table(patients);

    print("Columns Length: " + str(len(columns)) + " " + str(columns[:20]));

    for row in table:

        print("Patient Tuple Length: " + str(len(row)) + " " + str(row[:20]));

    return

if __name__ == "__main__":

    main(len(sys.argv), sys.argv)


#!/usr/bin/python3

import sys;

# return a bytearray from the file at the given path
def flatten_bin(path):

    eof1 = int();
    eof2 = int();

    result = list();
    infile = open(path, "rb");

    # prime the loop
    data = infile.read();

    # while there is content in the file
    while None != data:

        # for each byte in the input file
        for i in range(len(data)):
           
            eof2 = eof1;
            eof1 = data[i];

            # 0x00 0x3b is EOF...
            if eof2 == 0 and eof1 == 59:

                # mark the eof
                data = None;

                # trim spare null byte
                result = result[:len(result)];

                break;

            else:
            
                # add it to the result
                result.append(data[i]);

        if eof2 != 0 or eof1 != 59:

            # get the next bytes
            data = infile.read();

    # close the file
    infile.close();

    # return the result
    return result;


def main(argc, argv):

    flat = flatten_bin(argv[1]);

    for i in range(len(flat)):

        print(format(flat[i], "02x") + " ", end = "");

        if 0 == i % 20:

            print();

    print();

    return 0;

if __name__ == "__main__":

    if 2 > len(sys.argv):
        print("usage: " + str(sys.argv[0]) + " filename");

        exit();

    main(len(sys.argv), sys.argv);


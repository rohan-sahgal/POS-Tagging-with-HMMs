# The tagger.py starter code for CSC384 A4.
# Currently reads in the names of the training files, test file and output file,
# and calls the tagger (which you need to implement)
import os
import sys
import copy
import operator

def tag(training_list, test_file, output_file):
    # Tag the words from the untagged input file and write them into the output file.
    # Doesn't do much else beyond that yet.
    print("Tagging the file.")
    training_f = [open(t, "r") for t in training_list]
    
    test_f = open(test_file, "r")
    output_f = open(output_file, "w")

    # s = test_f.readline()[:-1] + " : NP0\n"
    # print(s)
    # output_f.write(s)
    
    #Build emission table

    wordcount_table = build_wordcount_table(training_f)

    emission_table = build_emission_table(wordcount_table)

    naive_tagger(emission_table, test_f, output_f)

def build_wordcount_table(training_f):

    word_dict = {}

    for f in training_f:
        for line in f.readlines():
            tup = line.split(":")

            if len(tup) == 3:
                word = ":"
                tag = tup[2].strip()
            else:
                word = tup[0].strip()
                tag = tup[1].strip()

            
            
            if word not in word_dict:
                word_dict[word] = {}
                word_dict[word][tag] = 1
            else:
                if tag not in word_dict[word]:
                    word_dict[word][tag] = 1
                else:
                    word_dict[word][tag] += 1

    return word_dict

def build_emission_table(wordcount_table):
    emission_dict = copy.deepcopy(wordcount_table)

    for word in emission_dict:
        if len(emission_dict[word]) >= 2:
            total = 0
            for tag in emission_dict[word]:
                total += emission_dict[word][tag]
            for tag in emission_dict[word]:
                emission_dict[word][tag] /= total
        else:
            for tag in emission_dict[word]:
                emission_dict[word][tag] = 1

    return emission_dict

def naive_tagger(emission_table, test_f, output_f):

    naive_choices = copy.deepcopy(emission_table)
    for word in naive_choices:
        naive_choices[word] = max(naive_choices[word].items(), key=operator.itemgetter(1))[0]
    
    for line in test_f.readlines():
        if line.strip() in naive_choices:
            s = line.strip() + " : " + naive_choices[line.strip()] + "\n"
            output_f.write(s)

if __name__ == '__main__':
    # Run the tagger function.
    print("Starting the tagging process.")

    # Tagger expects the input call: "python3 tagger.py -d <training files> -t <test file> -o <output file>"
    parameters = sys.argv
    training_list = parameters[parameters.index("-d")+1:parameters.index("-t")]
    test_file = parameters[parameters.index("-t")+1]
    output_file = parameters[parameters.index("-o")+1]
    print("Training files: " + str(training_list))
    print("Test file: " + test_file)
    print("Ouptut file: " + output_file)

    # Start the training and tagging operation.
    tag (training_list, test_file, output_file)
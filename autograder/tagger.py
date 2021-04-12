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
    

    word_tag_count, tag_transition_count, initial_tag_count = build_count_tables(training_f)

    emission_table = build_emission_table(word_tag_count)
    transition_table = build_transition_table(tag_transition_count)
    initial_table = build_initial_table(initial_tag_count)

    print(initial_table)

    naive_tagger(emission_table, test_f, output_f)

def naive_tagger(emission_table, test_f, output_f):

    print("Tagging naively, using only emission probabilities...")

    naive_choices = copy.deepcopy(emission_table)

    # Simply choose the highest emission probability for each word
    for word in naive_choices:
        naive_choices[word] = max(naive_choices[word].items(), key=operator.itemgetter(1))[0]
    
    for line in test_f.readlines():
        if line.strip() in naive_choices:
            s = line.strip() + " : " + naive_choices[line.strip()] + "\n"
            output_f.write(s)
        else:
            s = line.strip() + " : " + "???" + "\n"
            output_f.write(s)

def viterbi_tagger(observations, tags, initial_table, transition_table, emission_table):
    print("Tagging using the viterbi algorithm")



    # trellis = {}

    # for tag in tags:
    #     trellis[tag]

    # probability == p. Tm: the transition matrix. Em: the emission matrix.

    # function viterbi(observations, S, Π, Tm, Em): best_path 
    # trellis ← matrix(length(S), length(O))  # To hold p. of each state given each observation.
    # # Determine each hidden state's p. at time 0…
    # for s in range(length(S)):
    #     trellis[s, 0] ← Π[s] * Em[s, O[0]] 
    # # …and afterwards, assuming each state's most likely prior state, k.
    # for o in range(1, length(O)):
    #     for s in range(length(S)):
    #         k ← argmax(k in trellis[k, o-1] * Tm[k, s] * Em[s, o])
    #         trellis[s, o] ← trellis[k, o-1] * Tm[k, s] * Em[s, o]
    # best_path ← list()
    # for o in range(-1, -(length(O)+1), -1):  # Backtrack from last observation.
    #     k ← argmax(k in trellis[k, o])  # Most likely state at o 
    #     best_path.insert(0, S[k])       # is noted for return.
    # return best_path


def build_count_tables(training_f):

    print("Building intermediate structures...")

    word_tag_count = {}
    tag_transition_count = {}
    initial_tag_count = {}

    for f in training_f:
        prev_tag = None
        prev_word = None
        prev_prev_word = None

        opening_tag = False

        for line in f.readlines():
            
            tup = line.split(":")

            if len(tup) == 3:
                word = ":"
                tag = tup[2].strip()
            else:
                word = tup[0].strip()
                tag = tup[1].strip()

            if word == '"':
                opening_tag = not opening_tag

            # Logic for determining if a word is at the start of a sentence
            if (not prev_word) \
                or prev_word == "." and ( (word == '"' and opening_tag) or (word != '"' and word[0].isupper()) )\
                or prev_word == "?" and ( (word == '"' and opening_tag) or (word != '"' and word[0].isupper()) )\
                or prev_word == "!" and ( (word == '"' and opening_tag) or (word != '"' and word[0].isupper()) )\
                or prev_word == '"' and (word[0].isupper() or word == '"'):
                print(prev_prev_word, prev_word, word)
                if tag not in initial_tag_count:
                    initial_tag_count[tag] = 1
                else:
                    initial_tag_count[tag] += 1

            if prev_tag:
                if prev_tag not in tag_transition_count:
                    tag_transition_count[prev_tag] = {}
                    tag_transition_count[prev_tag][tag] = 1
                else:
                    if tag not in tag_transition_count[prev_tag]:
                        tag_transition_count[prev_tag][tag] = 1
                    else:
                        tag_transition_count[prev_tag][tag] += 1

            if word not in word_tag_count:
                word_tag_count[word] = {}
                word_tag_count[word][tag] = 1
            else:
                if tag not in word_tag_count[word]:
                    word_tag_count[word][tag] = 1
                else:
                    word_tag_count[word][tag] += 1

            prev_prev_word = prev_word
            prev_word = word
            prev_tag = tag

    return word_tag_count, tag_transition_count, initial_tag_count

def build_initial_table(initial_tag_count):
    initial_dict = copy.deepcopy(initial_tag_count)

    total = 0
    for tag in initial_dict:
        total += initial_dict[tag]
    for tag in initial_dict:
        initial_dict[tag] /= total
    
    return initial_dict

def build_transition_table(tag_transition_count):
    print("Building transition table...")
    # for entry in tag_transition_count:
    transition_dict = copy.deepcopy(tag_transition_count)

    for tag1 in transition_dict:
        if len(transition_dict[tag1]) >= 2:

            total = 0
            for tag2 in transition_dict[tag1]:
                total += transition_dict[tag1][tag2]
            for tag2 in transition_dict[tag1]:
                transition_dict[tag1][tag2] /= total
        else:
            for tag2 in transition_dict[tag1]:
                transition_dict[tag1][tag2] = 1

    return transition_dict

def build_emission_table(word_tag_count):
    print("Building emission table...")
    emission_dict = copy.deepcopy(word_tag_count)

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
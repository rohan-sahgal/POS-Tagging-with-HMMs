# The tagger.py starter code for CSC384 A4.
# Currently reads in the names of the training files, test file and output file,
# and calls the tagger (which you need to implement)
import os
import sys
import copy
import operator
import time

def tag(training_list, test_file, output_file):
    # Tag the words from the untagged input file and write them into the output file.
    # Doesn't do much else beyond that yet.
    print("Tagging the file.")
    training_f = [open(t, "r") for t in training_list]
    
    test_f = open(test_file, "r")
    output_f = open(output_file, "w")

    word_tag_count, tag_transition_count, initial_tag_count, total_tag_count, tag_word_count = build_count_tables(training_f)

    emission_table = build_emission_table(word_tag_count)
    transition_table = build_transition_table(tag_transition_count)
    initial_table = build_initial_table(initial_tag_count)
    reverse_emission_table = build_emission_table_reversed(tag_word_count)

    # naive_tagger(emission_table, total_tag_count, test_f, output_f)
    # viterbi_tagger(test_f, output_f, initial_table, transition_table, emission_table)
    new_viterbi_tagger(test_f, output_f, initial_table, transition_table, reverse_emission_table)

# Naive tagger, only uses emission probabilities
def naive_tagger(emission_table, total_tag_count, test_f, output_f):

    print("Tagging naively, using only emission probabilities...")

    most_common_tag = max(total_tag_count.items(), key=operator.itemgetter(1))[0]
    naive_choices = copy.deepcopy(emission_table)

    # Simply choose the highest emission probability for each word
    for word in naive_choices:
        naive_choices[word] = max(naive_choices[word].items(), key=operator.itemgetter(1))[0]
    
    for line in test_f.readlines():
        if line.strip() in naive_choices:
            s = line.strip() + " : " + naive_choices[line.strip()] + "\n"
            output_f.write(s)
        else:
            s = line.strip() + " : " + most_common_tag.strip() + "\n"
            output_f.write(s)

# Optimized Viterbi tagger, stores path more efficiently
def new_viterbi_tagger(test_f, output_f, initial_table, transition_table, emission_table):
    
    # Set up data structures
    observations = test_f.readlines()

    tag_list = []
    for tag in transition_table:
        tag_list.append(tag)
    
    prob_trellis = []
    path_trellis = []

    for i in range(len(tag_list)):
        prob_trellis.append([None] * len(observations))
        path_trellis.append([None] * len(observations))

    # Setting up the first column of the trellises
    i = 0
    first_obs = observations[0].strip()
    for tag in tag_list:
        
        if tag in emission_table and first_obs in emission_table[tag]:
            prob_trellis[i][0] = emission_table[tag][first_obs]
        else:
            prob_trellis[i][0] = 0.000000001
        
        if tag in initial_table:
            prob_trellis[i][0] *= initial_table[tag]
        else:
            prob_trellis[i][0] *= 0.000000001

        path_trellis[i][0] = -1
        i += 1
    
    # Iterate through columns of trellis
    j = 0  
    for observation in observations:

        if j == 0:
            j += 1
            continue
        
        stripped_obs = observation.strip()
        # Iterate through rows of trellis for this column
        k = 0
        total = 0
        for tag1 in tag_list:
            
            l = 0
            # Want to determine the state with the highest probability in the previous observation
            cur_max_prob = 0
            cur_prob = 0
            cur_max_index = 0
            cur_max_tag = None
            for tag2 in tag_list:
                cur_prob = prob_trellis[l][j - 1]
                if tag2 in transition_table and tag1 in transition_table[tag2]:
                    cur_prob *= transition_table[tag2][tag1]
                else:
                    cur_prob *= 0.000000001
                if tag1 in emission_table and stripped_obs in emission_table[tag1]:
                    cur_prob *= emission_table[tag1][stripped_obs]
                else:
                    cur_prob *= 0.000000001
                
                if cur_prob >= cur_max_prob:
                    cur_max_prob = cur_prob
                    cur_max_index = l
                    cur_max_tag = tag2

                l += 1
            
            max_prob = cur_max_prob
            max_tag_index = cur_max_index
            max_tag = cur_max_tag
            
            prob_trellis[k][j] = prob_trellis[max_tag_index][j - 1]
            if max_tag in transition_table and tag1 in transition_table[max_tag]:
                prob_trellis[k][j] *= transition_table[max_tag][tag1]
            else:
                prob_trellis[k][j] *= 0.000000001

            if tag1 in emission_table and stripped_obs in emission_table[tag1]:
                prob_trellis[k][j] *= emission_table[tag1][stripped_obs]
            else:
                prob_trellis[k][j] *= 0.000000001
        
            # Store the index of the best tag from the previous word in the path trellis
            path_trellis[k][j] = (max_tag_index)

            total += prob_trellis[k][j]

            k += 1
        
        # Normalize
        for m in range(len(prob_trellis)):
            prob_trellis[m][j] /= total

        j += 1

    # Get the maximum probability from the last column of the trellises
    # Construct dictionary for tags for more efficient reference when writing output
    tag_dict = {}    

    cur_max_prob = 0
    for i in range(len(tag_list)):

        tag_dict[i] = tag_list[i]

        if prob_trellis[i][len(observations) - 1] >= cur_max_prob:
            cur_max_prob = prob_trellis[i][len(observations) - 1]
            cur_max_index = i
        
    # Construct the path by tracing the path trellis backwards
    path = [cur_max_index]
    next_index = path_trellis[cur_max_index][len(observations) - 1]

    p = len(observations) - 1
    while next_index != -1:
        path.insert(0, next_index)
        next_index = path_trellis[next_index][p - 1]
        p -= 1

    # Using path, write tagged words to the output file
    for i in range(len(path)):
        s = observations[i].strip() + " : " + tag_dict[path[i]] + "\n"
        output_f.write(s)

# Initial implementation of Viterbi algorithm, slow and runs into memory errors with large test sets.
def viterbi_tagger(test_f, output_f, initial_table, transition_table, emission_table):

    observations = test_f.readlines()

    tag_list = []
    for tag in transition_table:
        tag_list.append(tag)
    
    prob_trellis = []
    path_trellis = []

    for i in range(len(tag_list)):
        prob_trellis.append([None] * len(observations))
        path_trellis.append([None] * len(observations))

    i = 0
    for tag in tag_list:
        if tag in initial_table:
            if observations[0].strip() in emission_table:
                if tag in emission_table[observations[0].strip()]:
                    prob_trellis[i][0] = initial_table[tag] * emission_table[observations[0].strip()][tag]
                else:
                    prob_trellis[i][0] = initial_table[tag] * 0.000000001
            else:
                prob_trellis[i][0] = initial_table[tag] * 0.000000001
        else:
            if observations[0].strip() in emission_table:
                if tag in emission_table[observations[0].strip()]:
                    prob_trellis[i][0] = 0.000000001 * emission_table[observations[0].strip()][tag]
                else:
                    prob_trellis[i][0] = 0.000000001
            else:
                prob_trellis[i][0] = 0.000000001
        path_trellis[i][0] = [i]
        i += 1
    
    # Iterate through columns of trellis
    probs = [None] * len(tag_list)
    j = 0  
    for observation in observations:

        if j == 0:
            j += 1
            continue
        
        # Iterate through rows of trellis for this column
        k = 0
        total = 0
        for tag1 in tag_list:
            
            l = 0
            # Want to determine the state with the highest probability in the previous observation
            
            for tag2 in tag_list:
                if tag2 in transition_table:
                    if tag1 in transition_table[tag2]:
                        if observation.strip() in emission_table:
                            if tag1 in emission_table[observation.strip()]:
                                probs[l] = ((prob_trellis[l][j - 1] * transition_table[tag2][tag1] * emission_table[observation.strip()][tag1], l, tag2))
                            else:
                                probs[l] = ((prob_trellis[l][j - 1] * transition_table[tag2][tag1] * 0.000000001, l, tag2))
                        else:
                            probs[l] = ((prob_trellis[l][j - 1] * transition_table[tag2][tag1] * 0.000000001, l, tag2))
                    else:
                        if observation.strip() in emission_table:
                            if tag1 in emission_table[observation.strip()]:
                                probs[l] = ((prob_trellis[l][j - 1] * 0.000000001 * emission_table[observation.strip()][tag1], l, tag2))
                            else:
                                probs[l] = ((prob_trellis[l][j - 1] * 0.000000001 * 0.000000001, l, tag2))
                        else:
                            probs[l] = ((prob_trellis[l][j - 1] * 0.000000001 * 0.000000001, l, tag2))
                else:
                    if observation.strip() in emission_table:
                        if tag1 in emission_table[observation.strip()]:
                            probs[l] = ((prob_trellis[l][j - 1] * 0.000000001 * emission_table[observation.strip()][tag1], l, tag2))
                        else:
                            probs[l] = ((prob_trellis[l][j - 1] * 0.000000001 * 0.000000001, l, tag2))
                    else:
                        probs[l] = ((prob_trellis[l][j - 1] * 0.000000001 * 0.000000001, l, tag2))
                l += 1

            max_tuple = max(probs, key=operator.itemgetter(0))
            max_tag_index = max_tuple[1]
            max_tag = max_tuple[2]
            
            if max_tag in transition_table:
                if tag1 in transition_table[max_tag]:
                    if observation.strip() in emission_table:
                        if tag1 in emission_table[observation.strip()]:
                            prob_trellis[k][j] = prob_trellis[max_tag_index][j - 1] * transition_table[max_tag][tag1] * emission_table[observation.strip()][tag1]
                        else:
                            prob_trellis[k][j] = prob_trellis[max_tag_index][j - 1] * transition_table[max_tag][tag1] * 0.000000001
                    else:
                        prob_trellis[k][j] = prob_trellis[max_tag_index][j - 1] * transition_table[max_tag][tag1] * 0.000000001
                else:
                    if observation.strip() in emission_table:
                        if tag1 in emission_table[observation.strip()]:
                            prob_trellis[k][j] = prob_trellis[max_tag_index][j - 1] * 0.000000001 * emission_table[observation.strip()][tag1]
                        else:
                            prob_trellis[k][j] = prob_trellis[max_tag_index][j - 1] * 0.000000001 * 0.000000001
                    else:
                        prob_trellis[k][j] = prob_trellis[max_tag_index][j - 1] * 0.000000001 * 0.000000001
            else:
                if observation.strip() in emission_table:
                    if tag1 in emission_table[observation.strip()]:
                        prob_trellis[k][j] = prob_trellis[max_tag_index][j - 1] * 0.000000001 * emission_table[observation.strip()][tag1]
                    else:
                        prob_trellis[k][j] = prob_trellis[max_tag_index][j - 1] * 0.000000001 * 0.000000001
                else:
                    prob_trellis[k][j] = prob_trellis[max_tag_index][j - 1] * 0.000000001 * 0.000000001
        
            path_trellis[k][j] = path_trellis[max_tag_index][j - 1] + [k]

            total += prob_trellis[k][j]

            k += 1
        
        # Normalize
        for m in range(len(prob_trellis)):
            prob_trellis[m][j] /= total

        j += 1

    z = 0

    tag_dict = {}    
    for i in range(len(tag_list)):
        tag_dict[i] = tag_list[i]

    for v in path_trellis[max_tag_index][j - 1]:
        s = observations[z].strip() + " : " + tag_dict[v] + "\n"
        print(s, end='')
        output_f.write(s)
        z += 1

# Builds the intermediate tables used for building initial, transition, and emission probability tables
def build_count_tables(training_f):

    print("Building intermediate structures...")

    word_tag_count = {}
    tag_word_count = {}
    tag_transition_count = {}
    initial_tag_count = {}
    total_tag_count = {}

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

                if tag not in initial_tag_count:
                    initial_tag_count[tag] = 1
                else:
                    initial_tag_count[tag] += 1

            if tag not in total_tag_count: 
                total_tag_count[tag] = 1
            else:
                total_tag_count[tag] += 1

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
            
            if tag not in tag_word_count:
                tag_word_count[tag] = {word: 1}
            else:
                if word not in tag_word_count[tag]:
                    tag_word_count[tag][word] = 1
                else:
                    tag_word_count[tag][word] += 1    
            
            prev_prev_word = prev_word
            prev_word = word
            prev_tag = tag

    return word_tag_count, tag_transition_count, initial_tag_count, total_tag_count, tag_word_count

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

# Builds emission table with tags as keys instead of words, used in new_viterbi_tagger
def build_emission_table_reversed(tag_word_count):
    print("Building reverse emission table...")
    reverse_emission_dict = copy.deepcopy(tag_word_count)

    for tag in reverse_emission_dict:
        if len(reverse_emission_dict[tag]) >= 2:
            total = 0
            for word in reverse_emission_dict[tag]:
                total += reverse_emission_dict[tag][word]
            for word in reverse_emission_dict[tag]:
                reverse_emission_dict[tag][word] /= total
        else:
            for word in reverse_emission_dict[tag]:
                reverse_emission_dict[tag][word] = 1

    return reverse_emission_dict

# Builds emission table with words as keys
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
    print("Starting the tagging process.")

    # Tagger expects the input call: "python3 tagger.py -d <training files> -t <test file> -o <output file>"
    parameters = sys.argv
    training_list = parameters[parameters.index("-d")+1:parameters.index("-t")]
    test_file = parameters[parameters.index("-t")+1]
    output_file = parameters[parameters.index("-o")+1]
    print("Training files: " + str(training_list))
    print("Test file: " + test_file)
    print("Output file: " + output_file)

    # Start the training and tagging operation.
    tag (training_list, test_file, output_file)
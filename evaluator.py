
import os
import sys

if __name__ == '__main__':
    # Invoke the shell command to train and test the HMM tagger

    parameters = sys.argv
    output_path = parameters[parameters.index("-o") + 1]
    solution_path = parameters[parameters.index("-s") + 1]

    # Compare the contents of the HMM tagger output with the reference solution.
    # Store the missed cases and overall stats in results.txt

    with open(output_path, "r") as output_file, \
         open(solution_path, "r") as solution_file, \
         open("results.txt", "w") as results_file:
        # Each word is on a separate line in each file.
        output = output_file.readlines()
        solution = solution_file.readlines()
        total_matches = 0

        # generate the report
        for index in range(len(output)):
            if output[index] != solution[index]:
                results_file.write(f"Line {index + 1}: "
                                   f"expected <{output[index].strip()}> "
                                   f"but got <{solution[index].strip()}>\n")
            else:
                total_matches = total_matches + 1

        # Add stats at the end of the results file.
        results_file.write(f"Total words seen: {len(output)}.\n")
        results_file.write(f"Total matches: {total_matches}.\n")
        results_file.write("Tagging accuracy: {:.2f}%.".format((total_matches / len(output)) * 100))
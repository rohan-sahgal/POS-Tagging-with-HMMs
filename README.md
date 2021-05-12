## Instructions

Run tagger.py with the training files (-d), the test files (-t), and the name of your output file (-o):

```
py tagger.py -d [training files] -t [test file] -o [output file]
```

The output will be a sequence of tagged words.

## Background

### POS Tagging

Part of speech tagging (also known as grammatical tagging) assigns grammatical tags to a sequence of words and punctuation symbols. It is used as a tool in Natural Language Processing, helping meaning to be discerned when a word has multiple possible meanings. This program and the following examples use the tags present in the [British National Corpus](http://www.natcorp.ox.ac.uk/docs/URG.xml).

&nbsp;

**_For example:_**

The word `John` is a proper noun, and should be assigned the tag `NP0`. 

The word `book` can be used as either a singular common noun or a finite/infinite form of verb, and can be assigned either the tag `NN1`, the tag `VVB`, or the tag `VVI`.

&nbsp;

There are many possible strategies to determine the appropriate tag for a word in a sequence. The strategy this program employs uses Hidden Markov Models to achieve that goal.

---

### Hidden Markov Models (HMMs)

Hidden Markov Models consist of two sequences of states: hidden states and observed states.

![image](https://user-images.githubusercontent.com/43156518/117897954-dc3e3b00-b291-11eb-8f1a-1e5df95d19b3.png)

For each observed state, there exists an underlying hidden state. Given a sequence of untagged words, the observed states would be the sequence of words, and the hidden states would be the POS tags for each word. We use probabilities to calculate what the most likely hidden state is for each given observed state. More specifically, we use three probability tables:

- Initial Probabilities: how likely it is for the first hidden state in a sequence to be each possible value
- Transition Probabilities: how likely it is for each hidden state to follow every other hidden state (including itself)
- Emission Probabilities: how likely each hidden state is given the observed state

These tables are calculated through training on existing sequences of tagged words.

Using these tables and a variation on the [Viterbi Algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm), we are able to determine the most likely POS tag for each word, and tag it accordingly.

<!---
"""This is the main program file for the auto grader program.
The auto grader assumes that the following files are in the same directory as the autograder:
  - autotraining.txt  -> file of tagged words used to train the HMM tagger
  - autotest.txt      -> file of untagged words to be tagged by the HMM
  - autosolution.txt  -> file with correct tags for autotest words
This auto grader generates a file called results.txt that records the test results.
"""
--->

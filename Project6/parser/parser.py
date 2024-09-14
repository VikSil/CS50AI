import nltk
import sys

from nltk.tokenize import RegexpTokenizer

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat"
V -> "smiled" | "tell" | "were"
"""

NONTERMINALS = """
S -> NP VP | S Conj S | S Conj VP
NP -> N | NP NP | Adj NP | Det NP | P NP | NP Adv
VP -> V | V NP | V Adj | Adv VP 
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    """
    sentence = sentence.lower()
    regex = RegexpTokenizer(r'\w+')
    return regex.tokenize(sentence)


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    NP_chunks = []

    # tree.subtrees() finds all trees at all levels,
    # i.e. it is already recursive
    for subtree in tree.subtrees():

        # can ignore VP because in case of V NP, that NP
        # will show up in the tree.subtrees() on its own
        if subtree.label() == 'NP':
            if not check_subtree(subtree):
                NP_chunks.append(subtree)

    return NP_chunks


def check_subtree(subtree: nltk.tree.tree.Tree) -> bool:
    """
    Function check if there are any NP tags
    one level deeper in a subtree
    """
    for node in subtree.subtrees():
        if subtree == node:
            continue

        if node.label() == 'NP':
            return True

    return False


if __name__ == "__main__":
    main()


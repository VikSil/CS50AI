import itertools
import csv
import sys

PROBS = {
    # Unconditional probabilities for having gene
    "gene": {2: 0.01, 1: 0.03, 0: 0.96},
    "trait": {
        # Probability of trait given two copies of gene
        2: {True: 0.65, False: 0.35},
        # Probability of trait given one copy of gene
        1: {True: 0.56, False: 0.44},
        # Probability of trait given no gene
        0: {True: 0.01, False: 0.99},
    },
    # Mutation probability
    "mutation": 0.01,
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {"gene": {2: 0, 1: 0, 0: 0}, "trait": {True: 0, False: 0}}
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (
                people[person]["trait"] is not None
                and people[person]["trait"] != (person in have_trait)
            )
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

        # one_gene = set(['Harry'])
        # two_genes = set(['James'])
        # have_trait = set(['James'])
        # p = joint_probability(people, one_gene, two_genes, have_trait)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (
                    True
                    if row["trait"] == "1"
                    else False if row["trait"] == "0" else None
                ),
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s)
        for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    people_list = list(people.keys())
    p = 1

    for person in people_list:

        mother = people[person]['mother']
        father = people[person]['father']
        parents = [parent for parent in [mother, father] if parent != None]
        p_parents = {mother: 1, father: 1}

        # probability of a parent to pass the gene on
        # multiply to the initial value of 1
        for parent in parents:
            if parent in one_gene:
                # 50% chance to pass the one gene on
                # both genes have the same probability
                # to mutate, so it cancels out
                p_parents[parent] *= 0.5

            elif parent in two_genes:
                # parent with two genes will pass it on
                # unless mutation occurs
                p_parents[parent] *= 1 - PROBS['mutation']

            else:
                # parent does not have genes
                # mutation must occur
                p_parents[parent] *= PROBS['mutation']

        p_gene = p_trait = 1

        # distributions for people with one gene
        if person in one_gene:

            # as per problem statement,
            # both parents are either known or unknown
            if mother == None:
                p_gene = PROBS['gene'][1]
            else:
                # add probabilities that one parent contributed and other did not
                p_gene = ((1 - p_parents[mother]) * p_parents[father]) + (
                    p_parents[mother] * (1 - p_parents[father])
                )

            p_trait = PROBS['trait'][1][person in have_trait]

        # distributions for people with two genes
        elif person in two_genes:

            # as per problem statement,
            # both parents are either known or unknown
            if mother == None:
                p_gene = PROBS['gene'][2]

            else:
                # multiply both parent's contribution, since they are concurrent
                p_gene = p_parents[mother] * p_parents[father]

            p_trait = PROBS['trait'][2][person in have_trait]

        # distributions for people with no genes
        else:

            # as per problem statement,
            # both parents are either known or unknown
            if mother == None:
                p_gene = PROBS['gene'][0]

            else:
                # multiply the probability that both parents
                # concurrently did not contribute
                p_gene = (1 - p_parents[mother]) * (1 - p_parents[father])

            p_trait = PROBS['trait'][0][person in have_trait]

        # probability that the gene profile and trait occurs simultaneously
        p *= p_gene * p_trait

    return p


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for person in list(probabilities.keys()):
        # update gene distribution
        if person in one_gene:
            probabilities[person]['gene'][1] += p
        elif person in two_genes:
            probabilities[person]['gene'][2] += p
        else:
            probabilities[person]['gene'][0] += p

        # update trait distribution
        if person in have_trait:
            probabilities[person]['trait'][True] += p
        else:
            probabilities[person]['trait'][False] += p


def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for person in list(probabilities.keys()):

        # normalise gene distribution
        zero = probabilities[person]['gene'][0]
        one = probabilities[person]['gene'][1]
        two = probabilities[person]['gene'][2]

        sum = zero + one + two

        probabilities[person]['gene'][0] = zero / sum
        probabilities[person]['gene'][1] = one / sum
        probabilities[person]['gene'][2] = two / sum

        # normalise trait distribution
        has = probabilities[person]['trait'][True]
        has_not = probabilities[person]['trait'][False]

        sum = has + has_not

        probabilities[person]['trait'][True] = has / sum
        probabilities[person]['trait'][False] = has_not / sum


if __name__ == "__main__":
    main()

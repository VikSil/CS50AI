import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(link for link in pages[filename] if link in pages)

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    N = len(corpus)
    distribution = dict.fromkeys(corpus)

    if len(corpus[page]) == 0:
        p = round(1 / len(corpus), 6)
        distribution = dict.fromkeys(distribution, p)

    else:
        p_random = (1 / N) * (1 - damping_factor)
        n_of_links = len(corpus[page])
        p_link = (1 / n_of_links) * damping_factor

        for key in distribution.keys():
            if key in corpus[page]:
                distribution[key] = round(p_link + p_random, 6)
            else:
                distribution[key] = round(p_random, 6)

    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    pages = list(corpus.keys())
    page = random.choice(pages)
    distribution = dict.fromkeys(pages, 0)

    for _ in range(n):
        distribution[page] += 1
        sample = transition_model(corpus, page, damping_factor)
        page = random.choices(list(sample.keys()), list(sample.values()), k=1)[0]

    for key, value in distribution.items():
        distribution[key] = round(value / n, 6)

    return distribution


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    pages = list(corpus.keys())
    N = len(pages)
    distribution = dict.fromkeys(pages, 1 / N)

    converge = False

    while not converge:
        new_distribution = dict.fromkeys(pages, 0)

        for page in pages:
            p_random = round((1 - damping_factor) / N, 6)

            # find all pages that link to page
            pages_i = []
            for page_i in corpus:
                if page in corpus[page_i] or len(corpus[page_i]) == 0:
                    pages_i.append(page_i)

            # calculate p-values
            sum = 0
            for page_i in pages_i:

                # if a page has no links, pretend it links to all pages
                n_i = len(corpus[page_i])
                if n_i == 0:
                    n_i = len(corpus)

                sum += distribution[page_i] / n_i

            p_linked = damping_factor * sum

            new_distribution[page] = p_random + p_linked

        # check if distribution has converged
        converge = True
        for page in pages:
            if abs(distribution[page] - new_distribution[page]) > 0.001:
                converge = False
                break

        distribution = new_distribution

    return distribution


if __name__ == "__main__":
    main()

import sys

from crossword import *


class CrosswordCreator:

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy() for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont

        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size, self.crossword.height * cell_size),
            "black",
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border, i * cell_size + cell_border),
                    (
                        (j + 1) * cell_size - cell_border,
                        (i + 1) * cell_size - cell_border,
                    ),
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (
                                rect[0][0] + ((interior_size - w) / 2),
                                rect[0][1] + ((interior_size - h) / 2) - 10,
                            ),
                            letters[i][j],
                            fill="black",
                            font=font,
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for var in self.domains:
            self.domains[var] = {
                value for value in self.domains[var] if len(value) == var.length
            }

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        revised = False
        no_matches = set()

        overlap = self.crossword.overlaps[x, y]
        if overlap is not None:
            for value in self.domains[x]:
                matches = {
                    match
                    for match in self.domains[y]
                    if value[overlap[0]] == match[overlap[1]]
                }
                if len(matches) == 0:
                    no_matches.add(value)
                    revised = True

        self.domains[x] = self.domains[x].difference(no_matches)

        return revised

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """

        if arcs is None:
            arcs = [(x, y) for x in self.domains for y in self.domains if x != y]

        while len(arcs) > 0:
            (x, y) = arcs.pop()
            if self.revise(x, y):
                if len(self.domains[x]) == 0:
                    return False

                for z in self.crossword.neighbors(x):
                    if z != y:
                        arcs.append((z, x))

        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """

        # calling a dict returns a list of its keys
        # then turn it into a set and compare sets
        if set(self.domains) == set(assignment):
            return True

        return False

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """

        assigned_values = assignment.values()

        # check if there are duplicate values
        if len(assigned_values) != len(set(assigned_values)):
            return False

        for x, value in assignment.items():
            # check that value is the correct length
            if x.length != len(value):
                return False

            # check that value has no conflicts
            for y in assignment:
                if x != y:
                    overlap = self.crossword.overlaps[x, y]
                    if overlap is not None:
                        if assignment[x][overlap[0]] != assignment[y][overlap[1]]:
                            return False

        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """

        domain = self.domains[var]

        # remove values that are already assigned
        for key in assignment:
            if assignment[key] in domain:
                domain = domain.difference(assignment[key])

        # least constraining values heuristic
        if len(domain) > 1:

            neighbors = self.crossword.neighbors(var)
            counter = {key: 0 for key in domain}

            for neighbor in neighbors:
                overlap = self.crossword.overlaps[var, neighbor]

                for value in domain:
                    for neighbour_value in self.domains[neighbor]:
                        if value[overlap[0]] != neighbour_value[overlap[1]]:
                            counter[value] += 1

                counter_sorted = dict(sorted(counter.items(), key=lambda item: item[1]))
                domain = list(counter_sorted)

        return domain

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """

        assigned_vars = set(assignment)
        all_vars = set(self.domains)
        unassigned_vars = all_vars.difference(assigned_vars)

        return_var = None

        # minimum remaining values heuristic
        for var in unassigned_vars:

            if return_var is None:
                return_var = var
            else:
                if len(self.domains[return_var]) > len(self.domains[var]):
                    return_var = var

                # degree heuristic
                elif len(self.domains[return_var]) == len(self.domains[var]):

                    degrees_return_var = self.crossword.neighbors(return_var)
                    degrees_var = self.crossword.neighbors(var)

                    if len(degrees_return_var) < len(degrees_var):
                        return_var = var

        return return_var

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """

        if self.assignment_complete(assignment):
            return assignment

        variable = self.select_unassigned_variable(assignment)
        new_assignment = assignment.copy()

        for value in self.order_domain_values(variable, new_assignment):
            if self.consistent(new_assignment):
                new_assignment[variable] = value
                result = self.backtrack(new_assignment)
                if result is not None:
                    return result

        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)

    # creator.revise(Variable(0, 2, "down", 3), Variable(0, 1, "across", 5))
    # print(creator.domains)

    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()

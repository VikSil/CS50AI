import random


class Minesweeper:
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence:
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        Returns the set of all cells in self.cells known to be mines.
        """
        if len(self.cells) == self.count:
            return self.cells

        return set()

    def known_safes(self):
        """
        Returns the set of all cells in self.cells known to be safe.
        """
        if self.count == 0:
            return self.cells

        return set()

    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.

        Returns True if cell was removed from the Sentence
        """
        if cell in self.cells:
            self.cells.remove(cell)
            self.count -= 1
            return True

        return False

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.

        Returns True if cell was removed from the Sentence
        """
        if cell in self.cells:
            self.cells.remove(cell)
            return True

        return False


class MinesweeperAI:
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """

        # mark the cell as a move that has been made
        self.moves_made.add(cell)

        # mark the cell as safe
        self.mark_safe(cell)

        # add a new sentences to the KB
        # based on the value of `cell` and `count`

        neighbours = set()

        for i in range(
            max(0, cell[0] - 1), min(self.height, cell[0] + 2)
        ):  # upper limit non-inclusive
            for j in range(max(0, cell[1] - 1), min(self.width, cell[1] + 2)):

                explore_cell = (i, j)

                # ignore the cell that called this function
                if explore_cell != cell:
                    # ignore neighbours that are already known to be safe
                    if explore_cell not in self.safes:
                        neighbours.add(explore_cell)

        new_sentence = Sentence(neighbours, count)
        self.knowledge.append(new_sentence)

        # mark any additional cells as safe or as mines
        self.update_cells()

        # make inferrences and add to the KB
        new_knowledge = self.make_knowledge()

        while new_knowledge:
            for sentence in new_knowledge:
                self.knowledge.append(sentence)

            self.housekeeping()
            self.update_cells()

            new_knowledge = self.make_knowledge()

    def housekeeping(self):
        """
        Removes any empty sets from the KB
        """
        pruned_knowledge = []
        for sentence in self.knowledge:
            if sentence not in pruned_knowledge:
                if len(sentence.cells) != 0:
                    pruned_knowledge.append(sentence)

        self.knowledge = pruned_knowledge

    def make_knowledge(self):
        """
        Makes new inferences from the knowledge already in the KB
        """
        inferences = []
        for sentence_a in self.knowledge:
            for sentence_b in self.knowledge:
                if sentence_a != sentence_b:  # don't compare sentence to itself
                    if sentence_a.cells.issubset(sentence_b.cells):
                        new_cells = sentence_b.cells.difference(sentence_a.cells)
                        new_count = sentence_b.count - sentence_a.count
                        inferred_sentence = Sentence(new_cells, new_count)
                        inferences.append(inferred_sentence)

        return inferences

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        for cell in self.safes:
            if cell not in self.moves_made:
                return cell

        return None

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        legal_moves = []

        for i in range(self.height):
            for j in range(self.width):
                cell = (i, j)
                if cell not in self.moves_made:
                    if cell not in self.mines:
                        legal_moves.append(cell)

        if len(legal_moves) != 0:
            return random.choice(legal_moves)

        return None

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)

        new_knowledge = False
        for sentence in self.knowledge:
            if sentence.mark_mine(cell):
                new_knowledge = True

        return new_knowledge

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)

        new_knowledge = False
        for sentence in self.knowledge:
            if sentence.mark_safe(cell):
                new_knowledge = True

        return new_knowledge

    def print_AI(self):
        """
        Outputs all sentences currently in the KB
        """
        print('THE MINES ARE:')
        print(self.mines)
        print('SAFES ARE:')
        print(self.safes)
        print('IT IS KNOWN THAT:')
        for sentence in self.knowledge:
            print(f'{sentence.cells} = {sentence.count}')

    def update_cells(self):
        """
        Distrubutes information about safe states and mines
        between the AI and KB
        """
        loop = True
        while loop:
            loop = False
            new_safes = set()
            new_mines = set()

            for sentence in self.knowledge:
                mines = sentence.known_mines()
                safes = sentence.known_safes()

                new_safes = new_safes | safes
                new_mines = new_mines | mines

            for cell in new_safes:
                self.mark_safe(cell)
                loop = True

            for cell in new_mines:
                self.mark_mine(cell)
                loop = True

            for cell in self.safes:
                if self.mark_safe(cell):
                    loop = True

            for cell in self.mines:
                if self.mark_mine(cell):
                    loop = True

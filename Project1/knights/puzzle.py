from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(
    # knowledge from problem setup
    Or(AKnave, AKnight),
    Not(And(AKnight, AKnave)),
    
    # knowledge from statement
    Implication(AKnave, Not(And(AKnave, AKnight))),
    Implication(AKnight, And(AKnight, AKnave)),
)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.
knowledge1 = And(
    # knowledge from problem setup
    Or(AKnave, AKnight),
    Not(And(AKnight, AKnave)),
    Or(BKnave, BKnight),
    Not(And(BKnight, BKnave)),
    
    # knowledge from statements
    Implication(AKnight, And(AKnave, BKnave)),
    Implication(AKnave, Not(And(AKnave, BKnave))),
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
knowledge2 = And(
    # knowledge from problem setup
    Or(AKnave, AKnight),
    Not(And(AKnight, AKnave)),
    Or(BKnave, BKnight),
    Not(And(BKnight, BKnave)),
    
    # knowledge from statements
    Implication(AKnight, Or(And(AKnight, BKnight), And(AKnave, BKnave))),
    Implication(AKnave, Not(Or(And(AKnight, BKnight), And(AKnave, BKnave)))),
    Implication(BKnight, Or(And(AKnight, BKnave), And(AKnave, BKnight))),
    Implication(BKnave, Not(Or(And(AKnight, BKnave), And(AKnave, BKnight)))),
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."
knowledge3 = And(
    # knowledge from problem setup
    Or(AKnave, AKnight),
    Not(And(AKnight, AKnave)),
    Or(BKnave, BKnight),
    Not(And(BKnight, BKnave)),
    Or(CKnave, CKnight),
    Not(And(CKnight, CKnave)),
    
    # knowledge from statements
    Or(  # A said one of the statements
        And(  # if A said "I am a knight."
            Implication(AKnight, AKnight), Implication(AKnave, Not(AKnight))
        ),
        And(  # or if A said "I am a knave."
            Implication(AKnight, AKnave), Implication(AKnave, Not(AKnave))
        ),
    ),
    Not(
        And(  # A did not say both of the above statements
            And(  # if A said "I am a knight."
                Implication(AKnight, AKnight), Implication(AKnave, Not(AKnight))
            ),
            And(  # or if A said "I am a knave."
                Implication(AKnight, AKnave), Implication(AKnave, Not(AKnave))
            ),
        )
    ),
    # B talking about what A said
    # and what A actually said, depending on who B and A are
    Implication(BKnight, Implication(AKnight, AKnave)),
    Implication(BKnight, Implication(AKnave, Not(AKnave))),
    # B talking about C
    Implication(BKnight, CKnave),
    Implication(BKnave, Not(CKnave)),
    # C talking about A
    Implication(CKnight, AKnight),
    Implication(CKnave, Not(AKnight)),
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3),
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()

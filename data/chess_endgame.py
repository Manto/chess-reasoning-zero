import os
import argparse
import chess
import chess.engine

# Load lichess_db_puzzle.csv into memory in a way that is easy to search and filter
import pandas as pd
from datasets import Dataset


def build_prompt(dp):
    puzzle = dp["puzzle"]
    return [
        {"role": "system", "content": "You are a helpful assistant. You first think about applying a reasoning process then provide the user with the answer."},
        {
            "role": "user",
            "content": f"You are a chess grandmaster. For this board configuration in FEN: {puzzle}, calculate the next move that will lead to checkmate, and tell me the move in UCI. Put your thought process in <think> </think> tag. Show your final answer in <answer> </answer> tag, for example <answer>b3b4</answer>",
        },
        {"role": "assistant", "content": "Let me solve this step by step. <think>"},
    ]


def search_mates(df, themes=None, min_move_count=4, max_move_count=8):
    """
    Search and filter the DataFrame for puzzles with the given themes and move counts.
    """
    filtered_df = df

    if themes is not None:
        filtered_df = filtered_df[
            filtered_df["Themes"].apply(
                lambda x: any(theme in x.lower() for theme in themes)
            )
        ]

    filtered_df = filtered_df[
        filtered_df["Moves"].apply(
            lambda x: len(x.split(" ")) >= min_move_count
            and len(x.split(" ")) <= max_move_count
        )
    ]

    return filtered_df


def make_puzzles(lichess_db_path):
    """
    Make a dataset of chess endgame puzzles and return a HF Dataset object
    Note: Get the puzzle database from https://database.lichess.org/lichess_db_puzzle.csv.zst
    """

    df = pd.read_csv(lichess_db_path)

    # Searching for puzzles involving a checkmate and a reasonable number of moves
    # TODO: Number of moves could be a hyperparameter to tune
    filtered_puzzles = search_mates(
        df, themes=["mate"], min_move_count=4, max_move_count=8
    )

    samples = {"puzzle": [], "solution": []}

    # Loop through each of the puzzles and print the FEN string
    for index, row in filtered_puzzles.iterrows():
        # Get FEN and Moves column from the row
        fen = row["FEN"]
        moves = row["Moves"].split(" ")

        # Initialize chess board with FEN string then apply the first move
        board = chess.Board(fen)
        move = chess.Move.from_uci(moves[0])
        board.push(move)

        # Print out the board FEN after the first move
        puzzle = board.fen()
        solution = moves[1]

        # Save puzzle, solution as a row in CSV
        samples["puzzle"].append(puzzle)
        samples["solution"].append(solution)

    dataset = Dataset.from_dict(samples)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="./chess")
    parser.add_argument("--lichess_db_path", default="./lichess_db_puzzle.csv")
    parser.add_argument("--train_size", type=int, default=327680)
    parser.add_argument("--test_size", type=int, default=1024)

    args = parser.parse_args()

    TRAIN_SIZE = args.train_size
    TEST_SIZE = args.test_size
    raw_dataset = make_puzzles(args.lichess_db_path)

    print(f"Total number of puzzles: {len(raw_dataset)}")

    assert len(raw_dataset) > TRAIN_SIZE + TEST_SIZE
    train_dataset = raw_dataset.select(range(TRAIN_SIZE))
    test_dataset = raw_dataset.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))

    def make_map_fn(split):
        def process_fn(example, idx):
            prompt = build_prompt(example)
            solution = {"puzzle": example["puzzle"], "solution": example["solution"]}
            data = {
                "data_source": "chess_endgame",
                "prompt": prompt,
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

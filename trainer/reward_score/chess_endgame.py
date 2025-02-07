# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import chess

def extract_solution(solution_str):
    """Extract the equation from the solution string."""
    # Remove everything before the first "Assistant:"
    if "Assistant:" in solution_str:
        solution_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        solution_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        return None
    solution_str = solution_str.split('\n')[-1]

    # Update the answer_pattern to be a chess UCI
     
    answer_pattern = r'<answer>([a-g][1-8][a-g][1-8])</answer>'
    match = re.finditer(answer_pattern, solution_str)
    matches = list(match)
    if matches:
        final_answer = matches[-1].group(1).strip()
    else:
        final_answer = None
    return final_answer

def get_score(answer, ground_truth):
    """
    Correct answer: 1.0
    Incorrect answer, but a valid move: 0.3
    Not a valid move, but fits format: 0.1
    """
    if not answer:
        return 0.0
    elif answer == ground_truth["solution"]:
        return 1.0
    else:
        board = chess.Board(ground_truth["puzzle"])
        if chess.Move.from_uci(answer) in board.legal_moves:
            return 0.2
        else:
            return 0.1

def compute_score(solution_str, ground_truth, method='strict', format_score=0., score=1.):
    """The scoring function for GSM8k.

    Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth
        format_score: the score for the format
        score: the score for the correct answer
    """

    answer = extract_solution(solution_str=solution_str)
    return get_score(answer, ground_truth)
        

if __name__ == '__main__':
    # Not a valid format at all
    assert compute_score("Assistant: The answer is <answer>blah</answer>", {"puzzle": "7k/p4r1p/3p1P2/3P2p1/1P2p3/1PQ5/3K1R1P/1q6 w - - 3 34", "solution": "c3c8"}) == 0

    # Could be a move, but not valid on the board
    assert compute_score("Assistant: The answer is <answer>c3b4</answer>", {"puzzle": "7k/p4r1p/3p1P2/3P2p1/1P2p3/1PQ5/3K1R1P/1q6 w - - 3 34", "solution": "c3c8"}) == 0.1

    # Is a move, but not checkmate sequence
    assert compute_score("Assistant: The answer is <answer>c3c4</answer>", {"puzzle": "7k/p4r1p/3p1P2/3P2p1/1P2p3/1PQ5/3K1R1P/1q6 w - - 3 34", "solution": "c3c8"}) == 0.2

    # Solution
    assert compute_score("Assistant: The answer is <answer>c3c8</answer>", {"puzzle": "7k/p4r1p/3p1P2/3P2p1/1P2p3/1PQ5/3K1R1P/1q6 w - - 3 34", "solution": "c3c8"}) == 1.0

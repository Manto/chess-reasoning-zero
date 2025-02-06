# chess-reasoning-zero

This is a reproduction of [DeepSeek R1 Zero](https://github.com/deepseek-ai/DeepSeek-R1) style training with chess endgame, built upon the work of [TinyZero](https://github.com/Jiayi-Pan/TinyZero) by Jiayi-Pan and team. The RL training is done with [veRL](https://github.com/volcengine/verl).

TinyZero has shown that with countdown and multiplication tasks, small models like Qwen2.5-3b is able to develop self-verification and search abilities and show measurable improvement over math evaluations requiring reasoning. This repository attemps to replicate similar capability via tasks of a different domain.

## Measuring Improvement Baseline

From the countdown tasks, running on 2x A100 80GB for xx steps, the experiment log and evaluation result are as follows:

...

## Chess Endgame Data point

Thanks to lichess.org for maintaining the chess puzzle database! Here's how you'd prepare the endgame dataset for training:

1. cd data
1. wget https://database.lichess.org/lichess_db_puzzle.csv.zst
1. unzstd lichess_db_puzzle.csv.zst
1. poetry run python chess_endgame.py

## Chess Endgame Training

...

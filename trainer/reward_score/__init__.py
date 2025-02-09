from verl.utils.reward_score import gsm8k, math
from trainer.reward_score import countdown, chess_endgame

def scoring_fn(data_source, solution_str, ground_truth):
    if data_source == 'openai/gsm8k':
        return gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in ['lighteval/MATH', 'DigitalLearningGmbH/MATH-lighteval']:
        return math.compute_score(solution_str, ground_truth)
    elif "countdown" in data_source:
        return countdown.compute_score(solution_str, ground_truth)
    elif "chess_endgame" in data_source:
        return chess_endgame.compute_score(solution_str, ground_truth)
    else:
        raise NotImplementedError
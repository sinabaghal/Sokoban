"""
Generate Sokoban puzzles using trained diffusion model.
"""

import argparse
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ModelConfig
from model import SokobanTransformer
from diffusion import MaskedDiffusion
from dataset import tensor_to_puzzle, CHAR_TO_ID


def validate_puzzle(puzzle: str) -> dict:
    """
    Validate a generated Sokoban puzzle.

    Returns dict with validation results.
    """
    lines = puzzle.strip().split('\n')
    result = {
        'valid': True,
        'errors': [],
        'stats': {}
    }

    # Count elements
    player_count = 0
    box_count = 0
    goal_count = 0
    box_on_goal = 0
    player_on_goal = 0

    for line in lines:
        for char in line:
            if char == '@':
                player_count += 1
            elif char == '$':
                box_count += 1
            elif char == '.':
                goal_count += 1
            elif char == '*':
                box_on_goal += 1
            elif char == '+':
                player_on_goal += 1

    total_players = player_count + player_on_goal
    total_boxes = box_count + box_on_goal
    total_goals = goal_count + box_on_goal + player_on_goal

    result['stats'] = {
        'players': total_players,
        'boxes': total_boxes,
        'goals': total_goals
    }

    # Validation rules
    if total_players != 1:
        result['valid'] = False
        result['errors'].append(f"Expected 1 player, found {total_players}")

    if total_boxes == 0:
        result['valid'] = False
        result['errors'].append("No boxes found")

    if total_boxes != total_goals:
        result['valid'] = False
        result['errors'].append(f"Boxes ({total_boxes}) != Goals ({total_goals})")

    return result


def generate_puzzles(
    checkpoint_path: str,
    num_samples: int = 10,
    strategy: str = 'random',
    device: str = 'cuda',
    validate: bool = True
):
    """Generate puzzles from trained model."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    config = ModelConfig()
    model = SokobanTransformer(config).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded checkpoint from step {checkpoint['step']}")
    print(f"Generating {num_samples} puzzles with {strategy} strategy...\n")

    # Generate
    diffusion = MaskedDiffusion(config)

    with torch.no_grad():
        samples = diffusion.sample(model, num_samples, device, strategy=strategy)

    # Display and validate
    valid_count = 0
    for i, sample in enumerate(samples):
        puzzle = tensor_to_puzzle(sample)
        print(f"{'='*40}")
        print(f"Puzzle {i+1}:")
        print(f"{'='*40}")
        print(puzzle)

        if validate:
            result = validate_puzzle(puzzle)
            status = "VALID" if result['valid'] else "INVALID"
            print(f"\nStatus: {status}")
            print(f"Stats: {result['stats']}")
            if result['errors']:
                print(f"Errors: {result['errors']}")
            if result['valid']:
                valid_count += 1
        print()

    if validate:
        print(f"{'='*40}")
        print(f"Valid puzzles: {valid_count}/{num_samples} ({100*valid_count/num_samples:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description='Generate Sokoban puzzles')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--num', type=int, default=10,
                        help='Number of puzzles to generate')
    parser.add_argument('--strategy', choices=['confidence', 'random'],
                        default='random', help='Sampling strategy')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--no-validate', action='store_true',
                        help='Skip puzzle validation')
    args = parser.parse_args()

    generate_puzzles(
        args.checkpoint,
        args.num,
        args.strategy,
        args.device,
        validate=not args.no_validate
    )


if __name__ == '__main__':
    main()

"""
Dataset for loading Sokoban puzzles.
"""

import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List

# Default data path (relative to this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, '..', '..', 'boxoban-levels', 'medium', 'train'))

# Token mapping
CHAR_TO_ID = {
    '#': 0,  # wall
    ' ': 1,  # floor
    '@': 2,  # player
    '$': 3,  # box
    '.': 4,  # goal
    '*': 5,  # box on goal
    '+': 6,  # player on goal
}

ID_TO_CHAR = {v: k for k, v in CHAR_TO_ID.items()}
MASK_TOKEN_ID = 7


def parse_boxoban_file(filepath: str) -> List[str]:
    """Parse a Boxoban file and return list of puzzles."""
    puzzles = []
    current_puzzle = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            if line.startswith(';'):
                if current_puzzle:
                    puzzles.append('\n'.join(current_puzzle))
                    current_puzzle = []
            elif line:
                current_puzzle.append(line)

        if current_puzzle:
            puzzles.append('\n'.join(current_puzzle))

    return puzzles


def puzzle_to_tensor(puzzle: str, height: int = 10, width: int = 10) -> torch.Tensor:
    """Convert puzzle string to tensor of token IDs."""
    lines = puzzle.strip().split('\n')

    # Pad to grid size
    grid = []
    for i in range(height):
        if i < len(lines):
            row = lines[i]
            # Pad row to width
            row = row.ljust(width)[:width]
        else:
            row = ' ' * width
        grid.append(row)

    # Convert to token IDs
    tokens = []
    for row in grid:
        for char in row:
            token_id = CHAR_TO_ID.get(char, 1)  # Default to floor
            tokens.append(token_id)

    return torch.tensor(tokens, dtype=torch.long)


def tensor_to_puzzle(tensor: torch.Tensor, height: int = 10, width: int = 10) -> str:
    """Convert tensor of token IDs back to puzzle string."""
    tokens = tensor.tolist()
    lines = []

    for i in range(height):
        row = ''
        for j in range(width):
            token_id = tokens[i * width + j]
            if token_id == MASK_TOKEN_ID:
                row += '?'
            else:
                row += ID_TO_CHAR.get(token_id, ' ')
        lines.append(row)

    return '\n'.join(lines)


class SokobanDataset(Dataset):
    """Dataset of Sokoban puzzles."""

    def __init__(self, data_path: str, height: int = 10, width: int = 10):
        self.height = height
        self.width = width
        self.puzzles = []

        # Load all puzzle files
        if os.path.isfile(data_path):
            files = [data_path]
        else:
            files = sorted(glob.glob(os.path.join(data_path, '*.txt')))

        for filepath in files:
            self.puzzles.extend(parse_boxoban_file(filepath))

        print(f"Loaded {len(self.puzzles)} puzzles from {len(files)} files")

    def __len__(self) -> int:
        return len(self.puzzles)

    def __getitem__(self, idx: int) -> torch.Tensor:
        puzzle = self.puzzles[idx]
        return puzzle_to_tensor(puzzle, self.height, self.width)


def create_dataloaders(
    train_path: str,
    val_path: str,
    batch_size: int,
    num_workers: int = 4
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""
    train_dataset = SokobanDataset(train_path)
    val_dataset = SokobanDataset(val_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == '__main__':
    # Test
    dataset = SokobanDataset(DEFAULT_DATA_PATH)
    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")
    print(f"Sample:\n{tensor_to_puzzle(sample)}")

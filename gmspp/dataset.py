"""
Dataset generation and persistence for GMSPP experiments.

Generate instances once, save to pickle, load everywhere.
This eliminates seeding/hashing issues across kernel restarts.
"""

import pickle
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from .data_structures import Item, Strip, Instance
from .instance_generator import generate_type1_instance


def generate_dataset(
    classes: List[int] = [1, 2, 3, 4],
    sizes: List[int] = [5, 8, 10, 12, 15],
    strip_configs: Dict[str, List[int]] = None,
    cost_types: List[str] = ['uniform'],
    n_seeds: int = 2,
    base_seed: int = 7000,
) -> Dict:
    """
    Generate a complete dataset of GMSPP instances.

    Returns a dict keyed by (class_id, n, strip_name, cost_type, seed_idx)
    mapping to Instance objects.

    Args:
        classes: MV class IDs
        sizes: number of items
        strip_configs: dict of strip_name -> list of widths
        cost_types: list of cost type strings
        n_seeds: number of seeds per configuration
        base_seed: starting seed

    Returns:
        dict mapping tuple keys to Instance objects
    """
    if strip_configs is None:
        strip_configs = {
            '2s': [50, 100],
            '3s': [30, 60, 100],
        }

    dataset = {}
    seed_counter = base_seed

    for cls in classes:
        for n in sizes:
            for sname, swidths in strip_configs.items():
                for cost_type in cost_types:
                    for idx in range(n_seeds):
                        key = (cls, n, sname, cost_type, idx)
                        inst = generate_type1_instance(
                            cls, n, swidths,
                            cost_type=cost_type,
                            seed=seed_counter,
                        )
                        dataset[key] = inst
                        seed_counter += 1

    return dataset


def save_dataset(dataset: Dict, filepath: str):
    """
    Save dataset to pickle file.

    Converts Instance objects to a serializable format.
    """
    serializable = {}
    for key, inst in dataset.items():
        serializable[key] = {
            'items': [(it.width, it.height) for it in inst.items],
            'strip_widths': [s.width for s in inst.strips],
            'strip_costs': [s.cost for s in inst.strips],
        }

    with open(filepath, 'wb') as f:
        pickle.dump(serializable, f)

    print(f'Saved {len(dataset)} instances to {filepath}')


def load_dataset(filepath: str) -> Dict:
    """
    Load dataset from pickle file.

    Reconstructs Instance objects from stored data.
    """
    with open(filepath, 'rb') as f:
        serializable = pickle.load(f)

    dataset = {}
    for key, data in serializable.items():
        items = [Item(id=i, width=w, height=h)
                 for i, (w, h) in enumerate(data['items'])]
        strips = [Strip(id=i, width=W, cost=c)
                  for i, (W, c) in enumerate(
                      zip(data['strip_widths'], data['strip_costs']))]
        dataset[key] = Instance(items=items, strips=strips)

    print(f'Loaded {len(dataset)} instances from {filepath}')
    return dataset


def generate_and_save(filepath: str, **kwargs) -> Dict:
    """Generate dataset and save to file. Returns the dataset."""
    dataset = generate_dataset(**kwargs)
    save_dataset(dataset, filepath)
    return dataset


def get_or_create_dataset(filepath: str, **kwargs) -> Dict:
    """Load dataset if exists, otherwise generate and save."""
    if Path(filepath).exists():
        return load_dataset(filepath)
    else:
        return generate_and_save(filepath, **kwargs)

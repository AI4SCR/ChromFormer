from ChromFormer.datasets import SyntheticDataset

from pathlib import Path
import pytest

# %%
def test_load_synthetic(tmp_path):
    tmp_path = Path(tmp_path)
    n_structures = 5
    ds = SyntheticDataset(path_save=tmp_path, n_structures=n_structures)
    assert len(ds.transfer_learning_hics) == n_structures
    assert len(ds.transfer_learning_structures) == n_structures
    assert len(ds.transfer_learning_distances) == n_structures

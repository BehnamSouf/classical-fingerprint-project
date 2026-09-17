from fingerprint_dataset.dataset import FingerprintDataset


def test_dataset_length():
    dataset = FingerprintDataset("data/FVC2002_DB2_B")
    assert len(dataset) == 80
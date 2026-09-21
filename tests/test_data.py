from modelopsforge.data import make_dataset, validate_dataset, FEATURES

def test_dataset_is_valid():
    dataset = make_dataset(samples=200)
    validate_dataset(dataset)
    assert list(dataset.X.columns) == FEATURES
    assert len(dataset.X) == 200

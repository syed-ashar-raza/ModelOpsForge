import argparse

from sklearn.model_selection import train_test_split

from .data import make_dataset, validate_dataset
from .evaluation import evaluate as evaluate_model
from .registry import load_champion
from .training import train_and_register


def train():
    result = train_and_register()
    print(result)


def evaluate():
    dataset = make_dataset()
    validate_dataset(dataset)

    _, X_test, _, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=0.2,
        random_state=42,
        stratify=dataset.y,
    )

    model = load_champion()
    metrics = evaluate_model(model, X_test, y_test)

    print(metrics.__dict__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "evaluate"])
    args = parser.parse_args()

    if args.command == "train":
        train()
    else:
        evaluate()

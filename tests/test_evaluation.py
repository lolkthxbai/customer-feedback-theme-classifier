import matplotlib.pyplot as plt

from src.evaluation import create_confusion_matrix_figure, evaluate_model


def test_evaluate_model_returns_labeled_confusion_matrix():
    labels = ["Checking or savings account", "Credit card"]
    results = evaluate_model(
        ["Credit card", "Checking or savings account", "Credit card"],
        ["Credit card", "Credit card", "Checking or savings account"],
        labels,
    )

    assert results["labels"] == labels
    assert results["confusion_matrix"].tolist() == [[0, 1], [1, 1]]

    figure = create_confusion_matrix_figure(results["confusion_matrix"], results["labels"])
    axis = figure.axes[0]

    assert axis.get_title() == "Confusion Matrix"
    assert axis.get_xlabel() == "Predicted category"
    assert axis.get_ylabel() == "Actual category"
    plt.close(figure)

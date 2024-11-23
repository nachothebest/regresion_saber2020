import math

import numpy as np

from model.predict import make_prediction


def test_make_prediction(sample_input_data):
    # Given
    expected_prediction_range = (0, 500)  # Define expected range for predictions
    expected_no_predictions = len(sample_input_data)  # Dynamic based on input size

    # When
    result = make_prediction(input_data=sample_input_data)
    print("Result from make_prediction:", result)  # Debugging output

    # Then
    predictions = result.get("predictions")
    assert predictions is not None, "Predictions are missing in the result."
    assert isinstance(predictions, list), "Predictions should be a list."


    # Ensure all predictions are numeric and within the expected range
    for pred in predictions:
        assert isinstance(pred, (float, np.floating)), f"Prediction {pred} is not numeric."
        assert expected_prediction_range[0] <= pred <= expected_prediction_range[1], (
            f"Prediction {pred} is out of range {expected_prediction_range}."
        )

    # Validate no errors occurred
    assert result.get("errors") is None, f"Errors found in result: {result.get('errors')}"

    # Check the number of predictions
    assert len(predictions) == expected_no_predictions, (
        f"Expected {expected_no_predictions} predictions, "
        f"but got {len(predictions)}."
    )


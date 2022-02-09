from evalutils import DetectionEvaluation
from evalutils.io import CSVLoader
from evalutils.validators import ExpectedColumnNamesValidator


class Node21_detection_baseline(DetectionEvaluation):
    def __init__(self):
        super().__init__(
            file_loader=CSVLoader(),
            validators=(
                ExpectedColumnNamesValidator(
                    expected=("image_id", "x", "y", "score")
                ),
            ),
            join_key="image_id",
            detection_radius=1.0,
            detection_threshold=0.5,
        )

    def get_points(self, *, case, key):
        """
        Converts the set of ground truth or predictions for this case, into
        points that represent true positives or predictions
        """
        try:
            points = case.loc[key]
        except KeyError:
            # There are no ground truth/prediction points for this case
            return []

        return [
            (p["x"], p["y"])
            for _, p in points.iterrows()
            if p["score"] > self._detection_threshold
        ]


if __name__ == "__main__":
    Node21_detection_baseline().evaluate()

def normalize_predictions(predictions: dict):
    """
    Accepts raw backend JSON and returns a normalized dict:
    {
      "pred_1x2": {...},
      "pred_ou25": {...},
      "pred_btts": {...}
    }
    Works with both formats:
      - prediction_1x2 / 1x2
      - prediction_ou25 / ou25
      - prediction_btts / btts
    """
    p1 = predictions.get("prediction_1x2") or predictions.get("1x2") or {}
    p25 = predictions.get("prediction_ou25") or predictions.get("ou25") or {}
    pb = predictions.get("prediction_btts") or predictions.get("btts") or {}

    return {
        "pred_1x2": p1,
        "pred_ou25": p25,
        "pred_btts": pb
    }

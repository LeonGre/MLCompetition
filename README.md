# MLCompetition

## Classification
how to make a Prediction:
- Python 3.12.1 is required to achieve the same results
- run SubmissionNr1.py/SubmissionNr2.py to calculate the predictions
- you can find the final predictions in classification-data/test_label.csv

---
## Regression
If you want to use the whole model make sure you updated smac to at least v2.0.2
how to make a Prediction:
- just start Models/ensemble_trainer.py
- have some time
Or:
- start the collaborative filtering model in predictions.py, by commenting out the model you want to use, to first calculate the predictions of this model
- start the Models/ensemble_predictions.py to ensemble it with another Regressor model
- you can find the final predictions in Models/final_predictions.csv

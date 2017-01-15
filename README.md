# RU-TextMining
Final assignment Text Mining
Joshua van Kleef

## Requirements
tensorflow 0.11
numpy
Flask

## Executing

``train.py`` needs to be run to train the model

``web.py`` starts a Flask web server were the trained model can be evaluated by calling ``http://localhost:5000/score?text=<text>`` and where <text> is the text that needs to be evaluated.

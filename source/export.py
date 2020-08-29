from joblib import dump
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


def export_model(score, classifier, x, y, filepath, study):
    try:
        if score > study.best_value:
            __export_models(classifier, x, y, filepath)
    except ValueError:
        __export_models(classifier, x, y, filepath)


def __export_models(classifier, x, y, filepath):
    classifier.fit(x, y)
    if type(classifier) == KerasClassifier:
        classifier.model.save(filepath)
    else:
        dump(classifier, filepath)

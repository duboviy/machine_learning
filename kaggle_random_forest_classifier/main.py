import os
import re
import argparse
import logging as log

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer


class ModelWrapper(object):

    def __init__(self, train_file, verify_file=None, evaluate_file=None, output_file=None,
                 model_type=RandomForestClassifier, n_estimators=200):
        self._train_file = train_file
        self._train_df = pd.read_json(train_file)
        self._verify_df = pd.read_json(verify_file) if verify_file else None
        self._evaluate_df = pd.read_json(evaluate_file) if evaluate_file else None
        self._output_file = output_file

        self._model_type = model_type
        self._n_estimators = n_estimators

        self._model = self._create_model()

        self._ingredients_tf = None
        self._cuisines_tf = None

        self._verify_result = None
        self._evaluate_result = None

    def _create_model(self):
        return self._model_type(self._n_estimators)

    def _train_model(self):
        """ Return: ingredients transformer, cuisines transformer """
        cuisines_tf, ingredients_tf = DictVectorizer(dtype=np.uint8), DictVectorizer(dtype=np.uint8)

        cuisines_tf.fit({r['cuisine']: 1} for _, r in self._train_df.iterrows())
        params = ingredients_tf.fit_transform(
            dict((self._norm_ingredient(i), 1) for i in r['ingredients']) for _, r in self._train_df.iterrows())
        outputs = np.fromiter((cuisines_tf.vocabulary_[r['cuisine']]
                               for _, r in self._train_df.iterrows()), np.uint8)

        self._model.fit(params, outputs)

        return ingredients_tf, cuisines_tf

    def _verify_model(self):
        """Verify trained model on specified data-frame"""
        verify_params = self._ingredients_tf.transform(
            dict((self._norm_ingredient(i), 1) for i in r['ingredients']) for _, r in self._verify_df.iterrows())
        verify_outputs = np.fromiter((self._cuisines_tf.vocabulary_[r['cuisine']]
                                      for _, r in self._verify_df.iterrows()), np.uint8)

        return self._model.score(verify_params, verify_outputs)

    def _evaluate_model(self):
        self._ingredients_tf, self._cuisines_tf = self._train_model()
        params = self._ingredients_tf.transform(
            dict((self._norm_ingredient(i), 1) for i in r['ingredients']) for _, r in self._evaluate_df.iterrows())

        return self._model.predict(params)

    def _save_eval_to_csv(self):
        self._evaluate_df['cuisine'] = [self._cuisines_tf.get_feature_names()[v] for v in self._evaluate_result]
        self._evaluate_df[['id', 'cuisine']].to_csv(self._output_file, index=False)

    @staticmethod
    def _norm_ingredient(ingredient):
        return " ".join(re.sub(r'\([^)]*\)', '', ingredient).lower().split())

    def verify(self):
        self._ingredients_tf, self._cuisines_tf = self._train_model()
        self._verify_result = self._verify_model()
        return self._verify_result

    def evaluate(self):
        self._ingredients_tf, self._cuisines_tf = self._train_model()
        self._evaluate_result = self._evaluate_model()
        self._save_eval_to_csv()

        return self._output_file, self._evaluate_result


def handler_verify(args):
    if not os.path.isfile(args.train_file):
        raise FileNotFoundError("Train file not found: {0}".format(args.file))

    if not os.path.isfile(args.verify_file):
        raise FileNotFoundError("Verify file not found: {0}".format(args.file))

    mv = ModelWrapper(args.train_file, verify_file=args.verify_file)
    result = mv.verify()

    log.info("Accuracy: {0:.5f}".format(result))


def handler_evaluate(args):
    if not os.path.isfile(args.train_file):
        raise FileNotFoundError("Train file not found: {0}".format(args.file))

    if not os.path.isfile(args.evaluate_file):
        raise FileNotFoundError("Evaluation file not found: {0}".format(args.file))

    mv = ModelWrapper(args.train_file, evaluate_file=args.evaluate_file, output_file=args.output_file)
    output_file, evaluate_result = mv.evaluate()

    log.info("Evaluated: %s", output_file)


def main():
    log.basicConfig(level=log.INFO, format='%(message)s')

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    verify_parser = subparsers.add_parser('verify')

    verify_parser.add_argument('-t', '--train-file', type=str, required=True, help='train file path')
    verify_parser.add_argument('-v', '--verify-file', type=str, required=True, help='verification file path')

    verify_parser.set_defaults(handler=handler_verify)

    evaluate_parser = subparsers.add_parser('evaluate')

    evaluate_parser.add_argument('-t', '--train-file', type=str, required=True, help='train file path')
    evaluate_parser.add_argument('-e', '--evaluate-file', type=str, required=True, help='evaluation file path')
    evaluate_parser.add_argument('-o', '--output-file', type=str, required=True, help='output file path')

    evaluate_parser.set_defaults(handler=handler_evaluate)

    args = parser.parse_args()
    args.handler(args)


if __name__ == '__main__':
    main()

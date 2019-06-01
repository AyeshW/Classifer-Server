from unittest import TestCase
from classifier import GeneralClassifier

class TestGeneralClassifier(TestCase):
    # noinspection PyCallByClass
    def test_classify(self):
        # deserialized = GeneralClassifier.deserialize()
        clf = GeneralClassifier()
        category = clf.classify('Sri Lanka cricket team won the 1996 world championship')
        ##assert.print("sport", category)
        self.assertEqual("sport", category)
        ## self().fail

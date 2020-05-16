"""
Subtokenizing-related tests.
"""
import os
import unittest

from ..subtokenizing import TokenParser

tests_dir = os.path.abspath(os.path.dirname(__file__))


class TestSubtokenizing(unittest.TestCase):
    test_subtokenizing_data = [["token", ["token"]],
                               ["Upper", ["upper"]],
                               ["camelCase", ["camel", "case"]],
                               ["snake_case", ["snake", "case"]],
                               ["os", []],
                               ["wdSize", ["size", "wdsize"]],
                               ["Egor.is.Nice", ["egor", "nice", "isnice"]],
                               ["stemming", ["stem"]],
                               ["sourced_directory", ["sourc", "directori"]],
                               ["some.ABSUrdSpecific_case.ml.in.code",
                                ["some", "abs", "urd", "specif", "case", "code", "incode"]]]

    def test_subtokenizer(self):
        subtokenizer = TokenParser(single_shot=False, min_split_length=3, stem_threshold=6)
        for data in TestSubtokenizing.test_subtokenizing_data:
            with self.subTest():
                subtokens = list(subtokenizer.process_token(data[0]))
                self.assertEqual(subtokens, data[1])


if __name__ == "__main__":
    unittest.main()

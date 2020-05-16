"""
Pipeline-related tests.
"""
import os
import unittest

from ..parsing import cmdline, recognize_languages, transform_files_list, transform_tokens

tests_dir = os.path.abspath(os.path.dirname(__file__))


class TestPipeline(unittest.TestCase):
    def test_cmdline(self):
        command = "echo 'Darina'"
        stdout = cmdline(command)
        self.assertEqual(stdout, "Darina\n")

    def test_languages(self):
        lang2files = recognize_languages(os.path.abspath(os.path.join(tests_dir, "test_files")))
        self.assertEqual(len(lang2files), 16)
        self.assertEqual(lang2files.keys(),
                         {"C", "C#", "C++", "Go", "Haskell", "Java", "JavaScript", "Kotlin", "PHP",
                          "Python", "Ruby", "Rust", "Scala", "Shell", "Swift", "TypeScript"})

    def test_transforming_list(self):
        lang2files = recognize_languages(os.path.abspath(os.path.join(tests_dir, "test_files")))
        files = transform_files_list(lang2files,
                                     os.path.abspath(os.path.join(tests_dir, "test_files")))
        self.assertEqual(len(files), 16)

    def test_transforming_tokens(self):
        tokens = [("height", 4), ("width", 4), ("rectangl", 2),
                  ("calc", 2), ("area", 2), ("prototyp", 1)]
        transformed_tokens = transform_tokens(tokens)
        self.assertEqual(transformed_tokens, ["height:4", "width:4", "rectangl:2",
                                              "calc:2", "area:2", "prototyp:1"])


if __name__ == "__main__":
    unittest.main()

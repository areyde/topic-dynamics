"""
Parsing-related functionality.
"""
from collections import Counter
import datetime
import json
from operator import itemgetter
import os
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, NamedTuple, Tuple

from joblib import cpu_count, delayed, Parallel
from pygments.lexers.haskell import HaskellLexer
from pygments.lexers.jvm import KotlinLexer, ScalaLexer
from pygments.lexers.objective import SwiftLexer
import pygments
from tqdm import tqdm
import tree_sitter

from .language_recognition.utils import get_enry
from .parsers.utils import get_parser
from .slicing import checkout_by_date, cmdline, get_dates, get_date_of_first_commit
from .subtokenizing import TokenParser

PROCESSES = cpu_count()

SliceLine = NamedTuple("SliceLine", [("date", str), ("start_index", int), ("end_index", int)])
TokenLine = NamedTuple("TokenLine", [("index", int), ("path", str), ("tokens", str)])

SUPPORTED_LANGUAGES = {"JavaScript": "tree-sitter",
                       "Python": "tree-sitter",
                       "Java": "tree-sitter",
                       "Go": "tree-sitter",
                       "C++": "tree-sitter",
                       "Ruby": "tree-sitter",
                       "TypeScript": "tree-sitter",
                       "TSX": "tree-sitter",
                       "PHP": "tree-sitter",
                       "C#": "tree-sitter",
                       "C": "tree-sitter",
                       "Scala": "pygments",
                       "Shell": "tree-sitter",
                       "Rust": "tree-sitter",
                       "Swift": "pygments",
                       "Kotlin": "pygments",
                       "Haskell": "pygments"}


class TreeSitterParser:
    PARSERS = {"JavaScript": "javascript",
               "Python": "python",
               "Java": "java",
               "Go": "go",
               "C++": "cpp",
               "Ruby": "ruby",
               "TypeScript": "typescript",
               "TSX": "tsx",
               "PHP": "php",
               "C#": "c_sharp",
               "C": "c",
               "Shell": "bash",
               "Rust": "rust"}

    NODE_TYPES = {"JavaScript": {"identifier", "property_identifier",
                                 "shorthand_property_identifier"},
                  "Python": {"identifier"},
                  "Java": {"identifier", "type_identifier"},
                  "Go": {"identifier", "field_identifier", "type_identifier"},
                  "C++": {"identifier", "namespace_identifier", "field_identifier",
                          "type_identifier"},
                  "Ruby": {"identifier", "constant", "symbol"},
                  "TypeScript": {"identifier", "property_identifier",
                                 "shorthand_property_identifier", "type_identifier"},
                  "TSX": {"identifier", "property_identifier",
                          "shorthand_property_identifier", "type_identifier"},
                  "PHP": {"name"},
                  "C#": {"identifier"},
                  "C": {"identifier", "field_identifier", "type_identifier"},
                  "Shell": {"variable_name", "command_name"},
                  "Rust": {"identifier", "field_identifier", "type_identifier"}}

    @staticmethod
    def read_file_bytes(file: str) -> bytes:
        """
        Read the contents of the file.
        :param file: the path to the file.
        :return: bytes with the contents of the file.
        """
        with open(file) as fin:
            return bytes(fin.read(), "utf-8")

    @staticmethod
    def get_positional_bytes(node: tree_sitter.Node) -> Tuple[int, int]:
        """
        Extract start and end byte of the tree-sitter Node.
        :param node: node on the AST.
        :return: (start byte, end byte).
        """
        start = node.start_byte
        end = node.end_byte
        return start, end

    @staticmethod
    def get_tokens(file: str, lang: str, subtokenizer: TokenParser) -> List[Tuple[str, int]]:
        """
        Gather a sorted list of identifiers in the file and their count.
        :param file: the path to the file.
        :param lang: the language of file.
        :param subtokenizer: TokenParser() with necessary parameters.
        :return: a list of tuples, identifier and count.
        """
        content = TreeSitterParser.read_file_bytes(file)
        tree = get_parser(TreeSitterParser.PARSERS[lang]).parse(content)
        root = tree.root_node
        tokens = []

        def traverse_tree(node: tree_sitter.Node) -> None:
            """
            Run down the AST (DFS) from a given node and gather tokens from its children.
            :param node: starting node.
            :return: None.
            """
            for child in node.children:
                if child.type in TreeSitterParser.NODE_TYPES[lang]:
                    start, end = TreeSitterParser.get_positional_bytes(child)
                    token = content[start:end].decode("utf-8")
                    if "\n" not in token:  # Will break output files.
                        subtokens = list(subtokenizer.process_token(token))
                        tokens.extend(subtokens)
                if len(child.children) != 0:
                    traverse_tree(child)

        try:
            traverse_tree(root)
        except RecursionError:
            return []
        return sorted(Counter(tokens).items(), key=itemgetter(1), reverse=True)


class PygmentsParser:
    LEXERS = {"Scala": ScalaLexer(),
              "Swift": SwiftLexer(),
              "Kotlin": KotlinLexer(),
              "Haskell": HaskellLexer()}

    TYPES = {"Scala": {pygments.token.Name, pygments.token.Keyword.Type},
             "Swift": {pygments.token.Name},
             "Kotlin": {pygments.token.Name},
             "Haskell": {pygments.token.Name, pygments.token.Keyword.Type}}

    @staticmethod
    def read_file(file: str) -> str:
        """
        Read the contents of the file.
        :param file: the path to the file.
        :return: the contents of the file.
        """
        with open(file) as fin:
            return fin.read()

    @staticmethod
    def get_tokens(file: str, lang: str, subtokenizer: TokenParser) -> List[Tuple[str, int]]:
        """
        Gather a sorted list of identifiers in the file and their count.
        :param file: the path to the file.
        :param lang: the language of file.
        :param subtokenizer: TokenParser() with necessary parameters.
        :return: a list of tuples, identifier and count.
        """
        content = PygmentsParser.read_file(file)
        tokens = []
        for pair in pygments.lex(content, PygmentsParser.LEXERS[lang]):
            if any(pair[0] in sublist for sublist in PygmentsParser.TYPES[lang]):
                tokens.extend(list(subtokenizer.process_token(pair[1])))
        return sorted(Counter(tokens).items(), key=itemgetter(1), reverse=True)


def parse_slice_line(slice_line: str) -> SliceLine:
    """
    Transform a line in the Slices file into the SliceLine format.
    :param slice_line: a line in the Slices file.
    :return: SliceLine object.
    """
    line_list = slice_line.rstrip().split(";")
    return SliceLine(line_list[0], int(line_list[1]), int(line_list[2]))


def parse_token_line(token_line: str) -> TokenLine:
    """
    Transform a line in Tokens file into the TokenLine format.
    :param token_line: a line in the Tokens file.
    :return: TokenLine object.
    """
    line_list = token_line.rstrip().split(";")
    return TokenLine(int(line_list[0]), line_list[1], line_list[2])


def recognize_languages(directory: str) -> dict:
    """
    Recognize the languages in the directory using Enry and return a dictionary
    {langauge1: [files], language2: [files], ...}.
    :param directory: the path to the directory.
    :return: dictionary {langauge1: [files], language2: [files], ...}
    """
    return json.loads(cmdline("{enry_loc} -json -mode files {directory}"
                              .format(enry_loc=get_enry(), directory=directory)))


def transform_files_list(lang2files: Dict[str, str], directory: str) -> List[Tuple[str, str]]:
    """
    Transform the output of Enry on a directory into a list of tuples (full_path_to_file, lang).
    :param lang2files: the dictionary output of Enry: {language: [files], ...}.
    :param directory: the full path to the directory that was processed with Enry.
    :return: a list of tuples (full_path_to_file, lang) for the supported languages.
    """
    files = []
    for lang in lang2files.keys():
        if lang in SUPPORTED_LANGUAGES.keys():
            for file in lang2files[lang]:
                files.append((os.path.abspath(os.path.join(directory, file)), lang))
    return files


def get_tokens(file: str, lang: str,
               subtokenizer: TokenParser) -> Tuple[str, List[Tuple[str, int]]]:
    """
    Gather a sorted list of identifiers in the file and their count.
    :param file: the path to the file.
    :param lang: the language of file.
    :param subtokenizer: TokenParser() with necessary parameters.
    :return: name of file, and a list of tuples, identifier and count.
    """
    if SUPPORTED_LANGUAGES[lang] == "tree-sitter":
        return file, TreeSitterParser.get_tokens(file, lang, subtokenizer)
    else:
        return file, PygmentsParser.get_tokens(file, lang, subtokenizer)


def transform_tokens(tokens: List[Tuple[str, int]]) -> List[str]:
    """
    Transform the original list of tokens into the writable form.
    :param tokens: list of tuples, token and count.
    :return: a list of tokens in the writable form, "token:count".
    """
    formatted_tokens = []
    for token in tokens:
        if token[0].rstrip() != "":  # Checking for occurring empty tokens.
            formatted_tokens.append("{token}:{count}".format(token=token[0].rstrip(),
                                                             count=str(token[1]).rstrip()))
    return formatted_tokens


def slice_and_parse(repositories_file: str, output_dir: str, dates: List[datetime.datetime],
                    single_shot: bool, min_token_length: int, min_stem_length: int) -> None:
    """
    Split the repository, parse the full files, write the data into a file.
    Can be called for parsing full files and for parsing diffs only.
    When run several times, overwrites the data.
    :param repositories_file: path to text file with a list of repositories to parse.
    :param output_dir: path to the output directory.
    :param dates: a list of dates used for slicing.
    :param single_shot: True for single-shot subtokenizing (not concatenating short subtokens),
                        False for concatenating short subtokens.
    :param min_token_length: any shorter subtoken will be either skipped or concatenated.
    :param min_stem_length: longer subtokens will be stemmed.
    :return: None.
    """
    print("Collecting the data about the repositories.")
    assert os.path.exists(repositories_file)
    repositories_list = []
    with open(repositories_file) as fin, \
            open(os.path.abspath(os.path.join(output_dir, "first_dates.txt")), "w+") as fout:
        for line in tqdm(fin):
            repositories_list.append([line.rstrip(), get_date_of_first_commit(line.rstrip())])
            assert os.path.exists(os.path.abspath(os.path.join(repositories_list[-1][0], ".git")))
            fout.write("{repository};{date}\n".format(repository=repositories_list[-1][0],
                                                      date=repositories_list[-1][1]))
    print("Creating the temporal slices of the data.")
    # Create a folder for created files
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    count = 0
    # Create temporal slices of the project, get a list of files for each slice,
    # parse all files, save the tokens
    subtokenizer = TokenParser(single_shot=single_shot, min_split_length=min_token_length,
                               stem_threshold=min_stem_length)
    with Parallel(PROCESSES) as pool, \
            open(os.path.abspath(os.path.join(output_dir, "tokens.txt")), "w+") as fout1, \
            open(os.path.abspath(os.path.join(output_dir, "slices.txt")), "w+") as fout2, \
            open(os.path.abspath(os.path.join(output_dir, "commits.txt")), "w+") as fout3, \
            open(os.path.abspath(os.path.join(output_dir, "bad_files.txt")), "w+") as fout4:
        for count_slice, date in enumerate(dates):
            print(f"Tokenizing slice {count_slice + 1} out of {len(dates)}.")
            start_index = count + 1
            fout3.write(date.strftime("%Y-%m-%d") + "\n\n")
            for repository in tqdm(repositories_list):
                if date > repository[1]:
                    with TemporaryDirectory() as td:
                        subdirectory = os.path.abspath(os.path.join(td, date.strftime("%Y-%m-%d")))
                        commit = checkout_by_date(repository[0], subdirectory, date)
                        fout3.write("{repository};{commit}\n".format(repository=repository[0],
                                                                     commit=commit))
                        lang2files = recognize_languages(td)
                        files = transform_files_list(lang2files, td)
                        chunk_results = pool([delayed(get_tokens)(file[0], file[1], subtokenizer)
                                              for file in files])
                        for chunk_result in chunk_results:
                            if (len(chunk_result[1]) != 0) and ("\n" not in chunk_result[0]) \
                                    and (";" not in chunk_result[0]):
                                count += 1
                                formatted_tokens = transform_tokens(chunk_result[1])
                                fout1.write("{file_index};{file_path};{tokens}\n"
                                            .format(file_index=str(count),
                                                    file_path=repository[0] + os.path.relpath(
                                                        os.path.abspath(os.path.join(
                                                            td, chunk_result[0])), td),
                                                    tokens=",".join(formatted_tokens)))
                            else:
                                fout4.write("{file_path}\n"
                                            .format(file_path=repository[0] + os.path.relpath(
                                                        os.path.abspath(os.path.join(
                                                            td, chunk_result[0])), td)))
            end_index = count
            if end_index >= start_index:  # Skips empty slices
                fout2.write("{date};{start_index};{end_index}\n"
                            .format(date=date.strftime("%Y-%m-%d"), start_index=str(start_index),
                                    end_index=str(end_index)))
            fout3.write("\n")


def split_token_file(slices_file: str, tokens_file: str, output_dir: str) -> None:
    """
    Split a single file with tokens into several by the date of the slice. For example, if the
    slices in the file are 2015-01-01, 2015-02-01, and 2015-03-01,
    it will divide the file into three.
    :param slices_file: the path to the file with the indices of the slices.
    :param tokens_file: the path to the temporary file with tokens.
    :param output_dir: path to the output directory.
    :return: None.
    """
    print("Splitting the tokens of full files by versions.")
    slice_number = 0
    date2indices = {}
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Read the data about the indices boundaries of slices
    with open(slices_file) as fin:
        for line in fin:
            slice_number = slice_number + 1
            slice_line = parse_slice_line(line)
            date2indices[slice_number] = (slice_line.start_index, slice_line.end_index)
    # Write the tokens of each slice into a separate file, numbered incrementally
    for date in tqdm(date2indices.keys()):
        with open(tokens_file) as fin, \
                open(os.path.abspath(os.path.join(output_dir, str(date) + ".txt")), "w+") as fout:
            for line in fin:
                token_line = parse_token_line(line)
                if (token_line.index >= date2indices[date][0]) and (
                        token_line.index <= date2indices[date][1]):
                    fout.write(line)


def read_tokens_counter(tokens: str) -> Counter:
    """
    Transform a string of tokens 'token1:count1,token2:count2' into a Counter object.
    :param tokens: input string of tokens 'token1:count1,token2:count2'.
    :return: Counter object of token tuples (token, count).
    """
    counter_tokens = Counter()
    for token_count in tokens.split(","):
        token, count = token_count.split(":")
        counter_tokens[token] = int(count)
    return counter_tokens


def differentiate_tokens(tokens: List[Tuple[str, int]], sign: str,
                         new_tokens: List[Any]) -> List[Tuple[str, int]]:
    """
    Transform the list of tuples (token, count) into the same list,
    but adding the necessary sign before each token (+ or -).
    :param tokens: input list of tuples (token, count).
    :param sign: sign of token, one of two: + or -.
    :param new_tokens: output list to append the results to.
    :return: list of differentiated tuples (+/-token, count).
    """
    assert sign in ["+", "-"]
    for token in tokens:
        new_token = sign + token[0]
        new_tokens.append([new_token, token[1]])
    return new_tokens


def calculate_diffs(slices_tokens_dir: str, output_dir: str,
                    dates: List[datetime.datetime]) -> None:
    """
    Given temporary tokens files of individual slices (separate files with tokens of each file for
    every slice), transform this data into a single tokens file with every slice except the first
    one, where for every slice and every file in it only changed tokens are saved: new tokens as
    '+token', deleted tokens as '-token'.
    :param slices_tokens_dir: the directory with token files split by slices.
    :param output_dir: path to the output directory.
    :param dates: a list of dates used for slicing.
    :return: None.
    """
    print("Calculating the diffs between versions and transforming the token lists.")
    count = 0
    with open(os.path.abspath(os.path.join(output_dir, "diffs_tokens.txt")), "w+") as fout1, \
            open(os.path.abspath(os.path.join(output_dir, "diffs_slices.txt")), "w+") as fout2:
        for date in tqdm(range(2, len(dates) + 1)):
            start_index = count + 1
            # Save the tokens of the "previous" slice into memory
            previous_version = {}
            with open(os.path.abspath(
                    os.path.join(slices_tokens_dir, str(date - 1) + ".txt"))) as fin:
                for line in fin:
                    token_line = parse_token_line(line)
                    previous_version[token_line.path] = token_line.tokens
            current_version = set()
            with open(os.path.abspath(os.path.join(slices_tokens_dir, str(date) + ".txt"))) as fin:
                for line in fin:
                    # Iterate over files in the "current" version
                    token_line = parse_token_line(line)
                    current_version.add(token_line.path)
                    tokens = read_tokens_counter(token_line.tokens)
                    old_path = token_line.path.replace(dates[date - 1].strftime("%Y-%m-%d"),
                                                       dates[date - 2].strftime("%Y-%m-%d"), 1)
                    # Check if the file with this name existed in the previous version
                    if old_path in previous_version.keys():
                        old_tokens = read_tokens_counter(previous_version[old_path])
                        # Calculate which tokens have been added and removed between versions
                        created_tokens = sorted((tokens - old_tokens).items(), key=itemgetter(1),
                                                reverse=True)
                        deleted_tokens = sorted((old_tokens - tokens).items(), key=itemgetter(1),
                                                reverse=True)
                        new_tokens = []
                        if len(created_tokens) != 0:
                            new_tokens = differentiate_tokens(created_tokens, "+", new_tokens)
                        if len(deleted_tokens) != 0:
                            new_tokens = differentiate_tokens(deleted_tokens, "-", new_tokens)
                    # If the file is new, all of its tokens are considered created
                    else:
                        tokens = sorted(tokens.items(), key=itemgetter(1), reverse=True)
                        new_tokens = []
                        new_tokens = differentiate_tokens(tokens, "+", new_tokens)
                    if len(new_tokens) != 0:
                        formatted_new_tokens = transform_tokens(new_tokens)
                        count = count + 1
                        fout1.write("{file_index};{file_path};{tokens}\n"
                                    .format(file_index=str(count),
                                            file_path=token_line.path,
                                            tokens=",".join(formatted_new_tokens)))
            # Iterate over files in the "previous" version to see which have been deleted
            for old_path in previous_version.keys():
                new_path = old_path.replace(dates[date - 2].strftime("%Y-%m-%d"),
                                            dates[date - 1].strftime("%Y-%m-%d"), 1)
                if new_path not in current_version:
                    tokens = sorted(read_tokens_counter(previous_version[old_path]).items(),
                                    key=itemgetter(1), reverse=True)
                    new_tokens = []
                    new_tokens = differentiate_tokens(tokens, "-", new_tokens)
                    formatted_new_tokens = transform_tokens(new_tokens)
                    count = count + 1
                    fout1.write("{file_index};{file_path};{tokens}\n"
                                .format(file_index=str(count),
                                        file_path=old_path,
                                        tokens=",".join(formatted_new_tokens)))
            end_index = count
            if end_index >= start_index:  # Skips empty slices
                fout2.write("{date};{start_index};{end_index}\n"
                            .format(date=dates[date - 1].strftime("%Y-%m-%d"),
                                    start_index=str(start_index), end_index=str(end_index)))


def uci_format(tokens_file: str, output_dir: str, name: str) -> None:
    """
    Transform the file with tokens into the UCI bag-of-words format. The format consists of two
    files: the first one lists all the tokens in the dataset alphabetically, and the second one
    lists all the triplets document-token-count, ranged first by documents, then by tokens.
    :param tokens_file: the path to the temporary file with tokens.
    :param output_dir: path to the output directory.
    :param name: name of the output dataset.
    :return: None.
    """
    print("Transforming the data into the UCI format for topic-modeling.")
    n_nnz = 0
    set_of_tokens = set()
    # Compile a list of all tokens in the dataset for a sorted list
    with open(tokens_file) as fin:
        for n_documents, line in enumerate(fin, start=1):
            token_line = parse_token_line(line)
            for token in token_line.tokens.split(","):
                n_nnz = n_nnz + 1
                set_of_tokens.add(token.split(":")[0])
    n_tokens = len(set_of_tokens)
    # Sort the list of tokens, transform them to indexes and write to file
    sorted_list_of_tokens = sorted(list(set_of_tokens))
    sorted_dictionary_of_tokens = {}
    with open(os.path.abspath(os.path.join(output_dir, "vocab." + name + ".txt")), "w+") as fout:
        for index in range(len(sorted_list_of_tokens)):
            sorted_dictionary_of_tokens[sorted_list_of_tokens[index]] = index + 1
            fout.write(sorted_list_of_tokens[index] + "\n")
    # Compile the second necessary file: NNZ triplets sorted by document
    with open(tokens_file) as fin, open(
            os.path.abspath(os.path.join(output_dir, "docword." + name + ".txt")), "w+") as fout:
        fout.write(str(n_documents) + "\n" + str(n_tokens) + "\n" + str(n_nnz) + "\n")
        for line in tqdm(fin):
            token_line = parse_token_line(line)
            file_tokens = token_line.tokens.split(",")
            file_tokens_separated = []
            file_tokens_separated_numbered = []
            for entry in file_tokens:
                file_tokens_separated.append(entry.split(":"))
            for entry in file_tokens_separated:
                file_tokens_separated_numbered.append(
                    [sorted_dictionary_of_tokens[entry[0]], int(entry[1])])
            file_tokens_separated_numbered = sorted(file_tokens_separated_numbered,
                                                    key=itemgetter(0), reverse=False)
            for entry in file_tokens_separated_numbered:
                fout.write("{doc_id} {token_id} {count}\n".format(doc_id=str(line.split(";")[0]),
                                                                  token_id=str(entry[0]),
                                                                  count=str(entry[1])))


def slice_and_parse_full_files(repository: str, output_dir: str, n_dates: int, day_delta: int,
                               single_shot: bool, min_token_length: int, min_stem_length: int,
                               start_date: str = None) -> None:
    """
    Split the repository, parse the full files, write the data into a file,
    transform into the UCI format.
    :param repository: path to the repository to process.
    :param output_dir: path to the output directory.
    :param n_dates: number of dates.
    :param day_delta: the number of days between dates.
    :param single_shot: True for single-shot subtokenizing (not concatenating short subtokens),
                        False for concatenating short subtokens.
    :param min_token_length: any shorter subtoken will be either skipped or concatenated.
    :param min_stem_length: longer subtokens will be stemmed.
    :param start_date: the starting (latest) date of the slicing, in the format YYYY-MM-DD,
    the default value is the moment of calling.
    :return: None.
    """
    dates = get_dates(n_dates, day_delta, start_date)
    tokens_file = os.path.abspath(os.path.join(output_dir, "tokens.txt"))
    slice_and_parse(repository, output_dir, dates, single_shot, min_token_length, min_stem_length)
    uci_format(tokens_file, output_dir, "dataset")
    print("Finished data preprocessing.")


def slice_and_parse_diffs(repository: str, output_dir: str, n_dates: int, day_delta: int,
                          single_shot: bool, min_token_length: int, min_stem_length: int,
                          start_date: str = None) -> None:
    """
    Split the repository, parse the full files, extract the diffs,
    write the data into a file, transform into the UCI format.
    :param repository: path to the repository to process.
    :param output_dir: path to the output directory.
    :param n_dates: number of dates.
    :param day_delta: the number of days between dates.
    :param single_shot: True for single-shot subtokenizing (not concatenating short subtokens),
                        False for concatenating short subtokens.
    :param min_token_length: any shorter subtoken will be either skipped or concatenated.
    :param min_stem_length: longer subtokens will be stemmed.
    :param start_date: the starting (latest) date of the slicing, in the format YYYY-MM-DD,
    the default value is the moment of calling.
    :return: None.
    """
    dates = get_dates(n_dates, day_delta, start_date)
    slices_file = os.path.abspath(os.path.join(output_dir, "slices.txt"))
    tokens_file = os.path.abspath(os.path.join(output_dir, "tokens.txt"))
    slices_tokens_dir = os.path.abspath(os.path.join(output_dir, "slices_tokens"))
    tokens_file_diffs = os.path.abspath(os.path.join(output_dir, "diffs_tokens.txt"))

    slice_and_parse(repository, output_dir, dates, single_shot, min_token_length, min_stem_length)
    split_token_file(slices_file, tokens_file, slices_tokens_dir)
    calculate_diffs(slices_tokens_dir, output_dir, dates)
    uci_format(tokens_file_diffs, output_dir, "diffs_dataset")
    print("Finished data preprocessing.")

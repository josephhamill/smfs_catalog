import time

import pytest

from smfs_catalog.path_pattern import path_matches

DNA = "/afm_data/2026/09/dna_run/Image0012.ibw"


@pytest.mark.parametrize("pattern", ["dna", "Image", "0012.ibw", "/09/", "DNA_RUN"])
def test_plain_text_is_a_case_insensitive_substring(pattern):
    assert path_matches(DNA, pattern)


def test_plain_text_that_is_absent_does_not_match():
    assert not path_matches(DNA, "rna")


def test_an_empty_pattern_constrains_nothing():
    assert path_matches(DNA, "")
    assert path_matches(DNA, None)
    assert path_matches(DNA, " , ")


@pytest.mark.parametrize("pattern", ["Image*.ibw", "Image00??.ibw", "*/dna_run/*"])
def test_wildcards_reach_the_filename_and_the_folders(pattern):
    assert path_matches(DNA, pattern)


def test_a_wildcard_spans_separators():
    assert path_matches(DNA, "2026*Image*")


def test_a_character_class_selects_one_position():
    assert path_matches("/afm_data/260106/x.ibw", "26010[6-8]")
    assert path_matches("/afm_data/260108/x.ibw", "26010[6-8]")
    assert not path_matches("/afm_data/260109/x.ibw", "26010[6-8]")


def test_a_character_class_is_one_character_not_a_number():
    # The standard's range spans characters, so 6-8 never reaches 18.
    assert not path_matches("/afm_data/260118/x.ibw", "2601[6-8]")


def test_a_negated_class_excludes():
    assert path_matches(DNA, "Image001[!0-1]")
    assert not path_matches(DNA, "Image001[!2]")


def test_commas_separate_patterns_and_any_one_may_match():
    assert path_matches(DNA, "rna_run,dna_run")
    assert not path_matches(DNA, "rna_run,protein_run")


def test_a_span_of_file_numbers_is_written_as_a_list_of_globs():
    span = "Image000[1-9],Image00[1-9][0-9],Image0[1-4][0-9][0-9],Image0500"
    for n in (1, 9, 10, 99, 100, 499, 500):
        assert path_matches(f"/data/Image{n:04d}.ibw", span)
    for n in (0, 501, 1000):
        assert not path_matches(f"/data/Image{n:04d}.ibw", span)


@pytest.mark.parametrize("pattern", ["a_b", "100%run"])
def test_sql_wildcards_no_longer_stand_for_anything(pattern):
    assert not path_matches("/afm_data/aXb/100_and_run/Image0001.ibw", pattern)


def test_a_missing_path_matches_nothing():
    assert not path_matches(None, "Image")
    assert not path_matches("", "Image")


@pytest.mark.parametrize("pattern", ["*?*?*?*?*?*?*?*z", "*a*a*a*a*a*b"])
def test_a_pathological_pattern_stays_cheap(pattern):
    # One row of a 10**5-row catalog: a millisecond here is minutes there.
    path = "/afm_data/Alexandre/260129-FC/ForceClamp0001.ibw"
    start = time.perf_counter()
    path_matches(path, pattern)
    assert time.perf_counter() - start < 0.05

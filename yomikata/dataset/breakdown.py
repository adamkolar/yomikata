from pathlib import Path
import pandas as pd
from yomikata.config import config, logger
from yomikata import utils
import re
from tqdm import tqdm
from collections import Counter
import gc
import os
import heapq

combinations_memo = {}


def find_combinations(surface, reading, df_grouped):
    if (surface, reading) not in combinations_memo:
        combinations_memo[(surface, reading)] = set()
        for i in range(len(surface)):
            surface_to_match = surface[:i+1]
            for candidate in df_grouped.get(surface_to_match, []):
                candidate_length = len(candidate['reading'])
                if candidate_length <= len(reading):
                    new_reading = reading[0]  # first character can't be 'ー'
                    for ii in range(1, candidate_length):
                        new_reading += candidate['reading'][ii] if reading[ii] == 'ー' and candidate['reading'][ii] in {'あ', 'え', 'い', 'お', 'う'} else reading[ii] # if possible, replace 'ー' with a vowel from the candidate reading
                    if new_reading == candidate['reading'] and (candidate['reading'] != reading or surface_to_match == surface): # candidate reading has to match part of the reading or whole reading if we're matching whole surface
                        remaining_surface = surface[i+1:]  # surface that hasn't been matched yet
                        remaining_reading = reading[candidate_length:]  # reading that hasn't been matched yet
                        if len(remaining_reading) == 0 or remaining_reading[0] != 'ー':  # if we've matched everything or unmatched reading doesn't start with 'ー'
                            combination_tuple = ((surface_to_match + '/' + candidate['reading'], candidate['count']),)
                            if len(remaining_surface) == 0 and len(remaining_reading) == 0:  # if we've matched everything we'll add the recently matched furigana and we're done
                                combinations_memo[(surface, reading)].add(combination_tuple)
                            elif len(remaining_surface) != 0 and len(remaining_reading) != 0:  # if we have both surface and reading left over, we continue
                                following_combinations = find_combinations(remaining_surface, remaining_reading, df_grouped)
                                if len(following_combinations) > 0:
                                    for combination in following_combinations:
                                        combinations_memo[(surface, reading)].add(combination_tuple + combination)  # we concatenate everything we've matched up until now with each candidate combination for the rest of the furigana
                                else:
                                    combinations_memo[(surface, reading)].add(combination_tuple + ((remaining_surface + '/' + remaining_reading, 1),))  # in case the remaining surface and reading doesn't match any known furigana, we just turn it into new furigana
        for i in range(len(surface)):  # same matching as above, but we match in reverse from the end of the furigana, this helps in case there's no good match for the start of the furigana
            surface_to_match = surface[-(i+1):]
            for candidate in df_grouped.get(surface_to_match, []):
                candidate_length = len(candidate['reading'])
                if candidate_length <= len(reading):
                    new_reading = reading[-candidate_length]
                    for ii in range(1, candidate_length):
                        new_reading += candidate['reading'][ii] if reading[-candidate_length + ii] == 'ー' and candidate['reading'][ii] in {'あ', 'え', 'い', 'お', 'う'} else reading[-candidate_length + ii]
                    if new_reading == candidate['reading'] and (candidate['reading'] != reading or surface_to_match == surface):
                        remaining_surface = surface[:-(i+1)]
                        remaining_reading = reading[:-candidate_length]
                        combination_tuple = ((surface_to_match + '/' + candidate['reading'], candidate['count']),)
                        if len(remaining_surface) == 0 and len(remaining_reading) == 0:
                            combinations_memo[(surface, reading)].add(combination_tuple)
                        elif len(remaining_surface) != 0 and len(remaining_reading) != 0:
                            preceding_combinations = find_combinations(remaining_surface, remaining_reading, df_grouped)
                            if len(preceding_combinations) > 0:
                                for combination in preceding_combinations:
                                    combinations_memo[(surface, reading)].add(combination + combination_tuple)
                            else:
                                combinations_memo[(surface, reading)].add(((remaining_surface + '/' + remaining_reading, 1),) + combination_tuple)
    return combinations_memo[(surface, reading)]


def generate_breakdown_dictionary():
    all_sentences_df = pd.read_csv(Path(config.SENTENCE_DATA_DIR, "all_filtered.csv"))
    sentences = all_sentences_df["furigana"].tolist()
    del all_sentences_df
    (translations, without_translation) = sentence_list_to_breakdown_dictionary(sentences)
    if not os.path.exists(config.BREAKDOWN_DATA_DIR):
        os.makedirs(config.BREAKDOWN_DATA_DIR)
    dict_path = Path(config.BREAKDOWN_DATA_DIR, "translations.json")
    utils.save_dict(translations, dict_path)
    logger.info("✅ Saved translation dictionary for decomposing furigana into " + str(dict_path) + "!")
    utils.save_dict(without_translation, Path(config.BREAKDOWN_DATA_DIR, "no_translations.json"))


def get_count(dictionary, string):
    first_letter = string[0]
    if first_letter not in dictionary:
        return False
    dictionary = dictionary[first_letter]
    if len(string) == 1:
        return dictionary["COUNT"]
    else:
        return get_count(dictionary, string[1:])


def equivalent_with_dash(dash_reading, other_reading):
    if len(dash_reading) != len(other_reading):
        return False
    for i in range(len(dash_reading)):
        if dash_reading[i] != other_reading[i] and (dash_reading[i] != 'ー' or other_reading[i] not in {'あ', 'え', 'い', 'お', 'う'}):
            return False
    return True


def find_dashless_equivalent(dash_combination, combinations):
    n_dashes = 0
    for furigana in dash_combination:
        (surface, reading) = furigana[0].split("/")
        n_dashes += reading.count("ー")
    if n_dashes == 0:
        return dash_combination
    best_combination = dash_combination
    for combination in combinations:
        n_dashes_comb = 0
        fits = True
        for i in range(len(combination)):
            furigana = combination[i]
            (surface, reading) = furigana[0].split("/")
            n_dashes_comb += reading.count("ー")
            (dash_surface, dash_reading) = dash_combination[i][0].split("/")
            if dash_surface != surface or not equivalent_with_dash(dash_reading, reading):
                fits = False
                break
        if fits and n_dashes_comb < n_dashes:
            best_combination = combination
            n_dashes = n_dashes_comb
    return best_combination


def get_highest_scoring_combination(combinations):
    """
    Score for a given furigana combination is a product of 
    (furigana frequency * furigana reading length) for each 
    furigana in the combination
    """
    max_len = 0
    max_score = 0
    best_combination = None
    for combination in combinations:
        lenght = len(combination)
        if lenght > max_len:
            max_len = lenght
            max_score = 0
        if lenght == max_len:
            score = 1
            for furigana in combination:
                (f_surface, f_reading) = furigana[0].split("/")
                count = furigana[1]
                score *= count * len(f_reading)
            if score > max_score:
                max_score = score
                best_combination = combination
    if best_combination is not None:
        best_combination = find_dashless_equivalent(best_combination, combinations)
    return best_combination


def sentence_list_to_breakdown_dictionary(sentences) -> dict:
    pattern = re.compile(r"\{((?:[^/{\}]|\{[^/\}]*\})*?)/([^/\}]*?[\u3040-\u309F\u30A0-\u30FF][^/\}]*?)\}")
    counter = Counter()
    for sentence in tqdm(sentences, desc="Compiling furigana"):
        matches = pattern.findall(sentence)
        counter.update(matches)  # we count all instances of each unique furigana in the corpus

    dictionary_df = pd.read_csv(Path(config.READING_DATA_DIR, "all.csv"))
    in_corpus = list(counter.keys())
    for idx in tqdm(dictionary_df.index, total=dictionary_df.shape[0], desc="Getting readings from unidic, sudachi and kanjidic2"):
        row = dictionary_df.loc[idx]
        counter[(row["surface"], row["kana"])] += 100  # give all dictionary furigana default score of 100
    del dictionary_df

    df_grouped = dict()

    for (surface, reading), count in tqdm(counter.items(), desc="Building furigana dictionary for the first pass"):
        if surface not in df_grouped:
            df_grouped[surface] = []
        df_grouped[surface].append({'reading': reading, 'count': count})

    for row in tqdm(in_corpus, desc="First pass to get implicit furigana fragments"):
        surface = row[0]
        reading = row[1]
        if len(surface) > 0:
            best_combination = get_highest_scoring_combination(find_combinations(surface, reading, df_grouped))
            if best_combination is not None:
                key = "{"+surface+"/"+reading+"}"
                if len(best_combination) != 1 or "{"+best_combination[0][0]+"}" != key:
                    for furigana in best_combination:
                        split_string = furigana[0].split("/")
                        counter.update([(split_string[0], split_string[1])])

    global combinations_memo
    combinations_memo = {}  # reset memoisation for find_combinations

    df_grouped = dict()
    for (kana, reading), count in tqdm(counter.items(), desc="Building furigana dictionary 2"):
        if kana not in df_grouped:
            df_grouped[kana] = []
        df_grouped[kana].append({'reading': reading, 'count': count})
    del counter

    translations = dict()
    without_translation = dict()

    for row in tqdm(in_corpus, desc="Breaking down furigana"):
        surface = row[0]
        reading = row[1]
        if len(surface) > 0:
            best_combination = get_highest_scoring_combination(find_combinations(surface, reading, df_grouped))
            key = "{"+surface+"/"+reading+"}"
            if best_combination != None and (len(best_combination) != 1 or "{"+best_combination[0][0]+"}" != key):
                translations[key] = ""
                for furigana in best_combination:
                    translations[key] += "{"+furigana[0]+"}"
            else:
                without_translation[key] = ""

    return (translations, without_translation)


if __name__ == "__main__":
    generate_breakdown_dictionary()
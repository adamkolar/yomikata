from pathlib import Path
import pandas as pd
from yomikata.config import config, logger
from yomikata import utils
import re
from tqdm import tqdm
from collections import Counter
import gc
import os

memo = {}
def find_combinations(surface, reading, df_grouped):
    if (surface, reading) not in memo:
        # print(surface, reading)
        memo[(surface, reading)] = []
        for i in range(len(surface)):
            for record in df_grouped.get(surface[:i+1], []):
                candidate_length = len(record['reading'])
                if candidate_length <= len(reading):
                    new_reading = reading[0]
                    for ii in range(1, candidate_length):
                        new_reading += record['reading'][ii] if reading[ii] == 'ー' else reading[ii]
                    # if new_reading.startswith(record['reading']):
                    if new_reading == record['reading'] and record['reading'] != reading:
                        remaining_surface = surface[i+1:]
                        remaining_reading = reading[candidate_length:]
                        record_tuple = [(surface[:i+1] + '/' + record['reading'], record['count'])]
                        if len(remaining_surface) == 0 and len(remaining_reading) == 0:
                            memo[(surface, reading)].append(record_tuple)
                        elif len(remaining_surface) != 0 and len(remaining_reading) != 0:
                            following_combinations = find_combinations(remaining_surface, remaining_reading, df_grouped)
                            if len(following_combinations) > 0:
                                for combination in following_combinations:
                                    memo[(surface, reading)].append(record_tuple + combination)
                            else:
                                memo[(surface, reading)].append(record_tuple + [(remaining_surface + '/' + remaining_reading, 0)])
    return memo[(surface, reading)]

def generate_breakdown_dictionary():
    all_sentences_df = pd.read_csv(Path(config.SENTENCE_DATA_DIR, "all.csv"))
    sentences = all_sentences_df["furigana"].tolist()
    del all_sentences_df
    # (translations, without_translation) = sentence_list_to_breakdown_dictionary(sentences)
    translations = sentence_list_to_breakdown_dictionary(sentences)
    if not os.path.exists(config.BREAKDOWN_DATA_DIR):
        os.makedirs(config.BREAKDOWN_DATA_DIR)
    dict_path = Path(config.BREAKDOWN_DATA_DIR, "translations.json")
    utils.save_dict(translations, dict_path)
    logger.info("✅ Saved translation dictionary for decomposing furigana into " + str(dict_path) + "!")
    # utils.save_dict(without_translation, Path(config.BREAKDOWN_DATA_DIR, "no_translations.json"))

def sentence_list_to_breakdown_dictionary(sentences) -> tuple[dict, dict]:
    # pattern = re.compile(r"\{(.*?)/(.*?)\}")
    pattern = re.compile(r"\{((?:[^/\}]|\{[^/\}]*\})*?)/([^/\}]*?[\u3040-\u309F\u30A0-\u30FF][^/\}]*?)\}")
    counter = Counter()
    for sentence in tqdm(sentences, desc="Compiling furigana"):
        matches = pattern.findall(sentence)
        counter.update(matches)
    gc.collect()
    dictionary_df = pd.read_csv(Path(config.READING_DATA_DIR, "all.csv"))
    in_corpus = list(counter.keys())
    for idx in tqdm(dictionary_df.index, total=dictionary_df.shape[0], desc="Getting readings from unidic and sudachi"):
        row = dictionary_df.loc[idx]
        counter.update([(row["surface"], row["kana"])])
    del dictionary_df
    gc.collect()
    df_grouped = dict()
    for (kana, reading), count in tqdm(counter.items(), desc="Building furigana dictionary"):
        if kana not in df_grouped:
            df_grouped[kana] = []
        df_grouped[kana].append({'reading': reading, 'count': count})
    del counter
    gc.collect()

    translations = dict()
    # without_translation = dict()

    for row in tqdm(in_corpus, desc="Breaking down furigana"):
        surface = row[0]
        reading = row[1]
        if len(surface) > 1:
            max_len = 0
            max_score = 0
            longest_combinations = []
            best_combination = None
            combinations = find_combinations(surface, reading, df_grouped)
            for combination in combinations:
                lenght = len(combination)
                if lenght > max_len:
                    max_len = lenght
                    max_score = 0
                if lenght == max_len:
                    score = 0
                    for furigana in combination:
                        score += furigana[1]
                    if score > max_score:
                        max_score = score
                        best_combination = combination
            key = "{"+surface+"/"+reading+"}"
            if best_combination != None:
                translations[key] = ""
                for furigana in best_combination:
                    translations[key] += "{"+furigana[0]+"}"
            # else:
            #     without_translation[key] = ""

    # return (translations, without_translation)
    return translations

if __name__ == "__main__":
    generate_breakdown_dictionary()

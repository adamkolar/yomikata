"""kanjidic.py
Data processing script for kanjidic2 dictionary xml files
Download: http://www.edrdg.org/kanjidic/kanjidic2.xml.gz
"""

import warnings
from pathlib import Path
import pandas as pd
from yomikata.config import config, logger
import xml.etree.ElementTree as ET
import jaconv

warnings.filterwarnings("ignore")


def kanjidic_data():
    kanjidic_file = list(Path(config.RAW_DATA_DIR, "kanjidic2").glob("*.xml"))
    df = pd.DataFrame()
    data = []
    for file in kanjidic_file:
        logger.info(file.name)
        tree = ET.parse(file)
        root = tree.getroot()
        for character in root.findall('character'):
            literal = character.find('literal').text
            reading_meaning = character.find('reading_meaning')
            if reading_meaning is not None:
                rmgroup = reading_meaning.find('rmgroup')
                if rmgroup is not None:
                    readings = rmgroup.findall('reading')
                    # Extract readings of type "ja_on" and "ja_kun"
                    for reading in readings:
                        if reading.get('r_type') in ['ja_on', 'ja_kun']:
                            data.append((literal, jaconv.hira2kata(reading.text)))

    df = pd.DataFrame(data, columns=['surface', 'kana'])
    df.to_csv(Path(config.READING_DATA_DIR, "kanjidic.csv"), index=False)
    logger.info("✅ Processed kanjidic2 data!")


if __name__ == "__main__":
    kanjidic_data()

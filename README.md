# Notebook changes / high-level overview

## Corpuses of annotated sentences
- added filtering of duplicate sentences at the end of "Corpuses of annotated sentences" section
## Spliting furigana in the corpus
- added "Spliting furigana in the corpus" section in the `yomikata.ipynb`
- added `yomikata/dataset/breakdown.py` with `generate_breakdown_dictionary` method
- added `yomikata.config.BREAKDOWN_DATA_DIR`
- added `yomikata.dataset.split.decompose_furigana`
## Making a list of heteronyms
- condensed the beginning of "Making a list of heteronyms" and included the list of heteronyms inline since it would be too complicated to compile the list programatically from the corpus in the notebook (we no longer have the `in_bert` and `readings` columns)
- removed the section filtering out overlapping heteronyms since we can newly handle them
- use `all_broken_down.csv` instead of `all.csv` since we work with sentences with the broken down furigana now
- use the `Counter` class from the native `collections` library instead of a custom function to simplify the code
## Process and split data
- drop filtering known sudachi compounds (`yomikata.dataset.split.filter_dictionary` function) since we observe better (or equal) general performance when leaving the compounds in the corpus
- also drop regrouping furigana (`yomikata.dataset.split.optimize_furigana` function) since that is no longer necessary because all furigana are broken down into smallest possible elements
## Dataset Info
- modify checking whether dataset sentences contain heteronyms slightly since dbert no longer works with `surfaceIDs` (label based on string matching not input id matching in `batch_preprocess_function`)
- simplify counting the occurences of heteronym readings in the corpus
## Train
- in `batch_preprocess_function`, add a `valid_mask` array to each token that defines which set of classes is valid for it based on the surface, this is used to restrict loss calculation during training only for these classes
- count occurences of each class in the corpus and apply weights to the loss function to compensate for some classes being underrepresented
- to implement a custom loss function we extend `BertForTokenClassification` with `CustomBertForTokenClassification`
- also implement `CustomDataCollatorForTokenClassification` to pad our `valid_mask` arrays to the sequence length
- change how labeling works
	- instead of adding new tokens for long heteronyms, label the first token in the heteronym
	- instead of identifying relevant tokens based on matching their id, locate heteronyms in the sentence strings and then identify tokens corresponding with the starting location of the heteronyms
- modify `dbert.furigana` to identify relevant tokens based on substring matching instead of input id matching and only consider logits of classes that are valid for a given surface
- get rid of the `<OTHER>` class, instead simply skip labelling a token if the reading isn't in the set of readings we predict for, this is consistent with sentences containing only invalid readings being dropped during dataset pre-processing
## Use, Code structure, Test on datasets, Performance for dictionar
- adapted these sections to make them work with new corpus format and the new reader

# Modified files:
* yomikata/notebooks/yomikata.ipynb
* yomikata/dbert.py
	* changed `metric_for_best_model` to `accuracy` and added efficient accuracy calculation to the training process
	+ added focal loss
	+ added masking out logits for irrelevant classes in loss calculation
	* only considering predictions for relevant classes in the `furigana` method and performance evaluation
	* some quality of life improvements like automatic reinitialisation of dbert on first run, easier configuration etc.
+ yomikata/custom_bert.py
	+ add `CustomBertForTokenClassification` and `CustomDataCollatorForTokenClassification` to implement masking out irrelevant logits and focal loss (https://paperswithcode.com/method/focal-loss)
* yomikata/config/config.py
	+ added `BREAKDOWN_DATA_DIR` to store dictionary for furigana breakdown
+ yomikata/dataset/breakdown.py
	+ added `generate_breakdown_dictionary` which generates a file with translations of larger furigana to a series of shorter ones based on furigana present in a corpus
	+ added `sentence_list_to_breakdown_dictionary` which takes a list of sentences with furigana and returns a dictionary of long furigana and roken down representations
* yomikata/dataset/split.py
	* changed `remove_other_readings` updated to make it work even if single furigana doesn't exactly match the valid reading, necessary because we're not regrouping furigana anymore
	+ added `decompose_furigana` which takes an input file with sentences and a furigana breakdown dictionary and outputs a file with the furigana broken down
	* changed `if __name__ == "__main__"`:
* yomikata/utils.py
	* `LabelEncoder` (added group_boundaries for the purpose of masking logits of irrelevant classes)
	+ `get_all_surface_readings`
	+ `find_all_substrings`
	+ `get_furis`
	+ `get_reading_from_furi`
* requirements
	* bumped up some versions to solve compatibility issues
	+ added `jupyter`
	+ added `evaluate`
	+ added `accelerate`
* yomikata/dictionary.py
	* `furigana` : only keep furigana from the `text` argument if it exactly matches one of the tokens from the tagger, this is done so that compound readings from the dictionary are retained
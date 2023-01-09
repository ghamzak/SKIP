# SKIP
Pytorch implementation of **S**emantic **K**nowledge **I**n **P**rocedural text understanding (SKIP) model on [ProPara dataset](https://allenai.org/data/propara). The SKIP model was heavily inspired by [KOALA model](https://github.com/ytyz1307zzh/KOALA) developed by researchers from Microsoft and Peking Uniersity (See the [leaderboard](https://leaderboard.allenai.org/propara/submissions/public)). The main difference of SKIP and KOALA is that KOALA only uses commonsense knowledge extracted from ConceptNet, while SKIP uses semantic knowledge as well. In particular, SKIP uses the semantic predicates extracted from the VerbNet semantic parse of each sentence in the dataset, using the information encoded in the semantic predicates for informing the model about the semantics of the event that is happenning to individual entities in a given sentence.

## Data

SKIP is designed to use and solve the challenge proposed by the [ProPara dataset](http://data.allenai.org/propara/) created by AI2. This dataset is about a machine reading comprehension (MRC) task on procedural text, *i.e.*, a text paragraph that describes a natural process (*e.g.*, photosynthesis, evaporation, etc.). AI models are required to read the paragraph, then predict the state changes for a numeber of entities, including the change of state types CREATE, MOVE, DESTROY or NONE, as well as the locations of the given entities before and after the change of state (see the [original paper](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://arxiv.org/pdf/1805.06975.pdf)).

AI2 released the dataset [here](https://docs.google.com/spreadsheets/d/1x5Ct8EmQs2hVKOYX7b2nS0AOoQi4iM7H9d9isXRDwgM/edit#gid=832930347) in the form of a Google Spreadsheet. We need three files to run the SKIP model, *i.e.*, the Paragraphs file for the raw text, the Train/Dev/Test file for the dataset split, and the State_change_annotations file for the annotated entities and their locations. I also provide a copy in `data/` directory which is identical to the official release.

## Setup

1. Create a virtual environment with python >= 3.8.

2. Install the dependency packages in `requirements.txt`:

   ```bash
   pip install -r requirements.txt
   ```

3. If you want to create your own dataset using `preprocess.py`, you also need to download the en_core_web_sm model for English language support of SpaCy:

   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

1. [Download](https://docs.google.com/spreadsheets/d/1x5Ct8EmQs2hVKOYX7b2nS0AOoQi4iM7H9d9isXRDwgM/edit#gid=832930347) the dataset or use my copy in `data/`.

2. Every single sentence in the dataset was parsed using the [VerbNet parser](https://github.com/jgung/verbnet-parser). If you want to perform parsing on your own, or on a different dataser, refer to this link for more information. Otherwise, I have stored a copy of the parsed files [here](https://drive.google.com/drive/folders/12cCyLme4ON_ns4n4KYMVI2OmdLmaOT0z?usp=sharing).

3. Process the CSV data files:

   ```bash
   python preprocess.py
   ```

   By default, the files should be put in `data/` and the output JSON files are also stored in `data/`. You can specify the input and output paths using optional command-line arguments. Please refer to the code for more details of command-line arguments.
   
   **P.S.** Please download the files in CSV format if you want to process the raw data using `preprocess.py`.
   
   **P.P.S.** If you choose to preprocess your own data, your dataset will get different orders of location candidates (compared to my copy in `data/`) due to the randomness of python set. This will lead to slightly different model performance due to the existence of dropout.
   
4. **Knowledge preparation**: To reproduce SKIP, you need to format knowledge triples such that it can be fed into a neural model. cd to the `Conceptnet` directory, where all the external knowledge preparation and preprocessing happen, and do the following:
   a. run `rough_retrieval_vn.py` 
   b. run `translate_vn.py`
   c. run `combine_vn_cn.py`

The output is the file `retrieval.json` under `Conceptnet/result/`.

5. The BERT encoder fine-tuned on additional Wiki paragraphs are stored on Google Drive, [here](https://drive.google.com/drive/folders/1jAoy093PSleMiRtk8vwDU75qpYWxe00t?usp=share_link).

5. Train a SKIP model:

   ```bash
   python train.py -mode train -ckpt_dir ckpt -train_set data/train.json -dev_set data/dev.json\
   -cpnet_path CPNET_PATH -cpnet_plm_path CPNET_PLM_PATH -cpnet_struc_input -state_verb STATE_VERB_PATH\
   -wiki_plm_path WIKI_PLM_PATH -finetune
   ```

   where `-ckpt_dir` denotes the directory where checkpoints will be stored, `-train_set` and `-dev_set` denote the preprocessed train and dev splits, 

`-cpnet_path` denites the json file containing all types of symbolic knowledge that we want to feed into the model while training, and `-state_verb` should point to the co-appearance verb set of entity states. My copy of these two files are stored in `ConceptNet/result/`. Please refer to the README file under `ConceptNet/` directory for more details of generating these files from scratch.

   `-cpnet_plm_path` denotes the directory in which the results of fine-tuning a BERT encoder on the symbolic knowledge of your choice is stored, and `-wiki_plm_path` denotes the directory in which the results of fine-tuning a BERT encoder on additional Wikipedia articles containing procedural texts is stored. My copy of these fine-tuned encoders are stored in [Google Drive](https://drive.google.com/drive/folders/1-PZXLGRBAY_1B4ANKqC93wqUOwVwQ95U?usp=sharing). Please refer to the README files under `wiki/` and `finetune/` directories for more details of collecting Wiki paragraphs and fine-tuning BERT encoders.

   Some useful training arguments:

   ```
   -save_mode     Checkpoint saving mode. 'best' (default): only save the best checkpoint on dev set. 
                  'all': save all checkpoints. 
                  'none': don't save checkpoints.
                  'last': save the last checkpoint.
                  'best-last': save the best and the last checkpoints.
   -epoch         Number of epochs to run the dataset. You can set it to -1 
                  to remove epoch limit and only use early stopping 
                  to stop training.
   -impatience    Early stopping rounds. If the accuracy on dev set does not increase for -impatience rounds, 
                  then stop the training process. You can set it to -1 to disable early stopping 
                  and train for a definite number of epochs.
   -report        The frequency of evaluating on dev set and save checkpoints (per epoch).
   ```

   Time for training a new model may vary according to your GPU performance as well as your training schema (*i.e.*, training epochs and early stopping rounds). 

5. Predict on test set using a trained model:

   ```bash
   python -u train.py -mode test -test_set data/test.json -dummy_test data/dummy-predictions.tsv\
   -output predict/prediction.tsv -cpnet_path CPNET_PATH -cpnet_plm_path CPNET_PLM_PATH\
   -cpnet_struc_input -state_verb STATE_VERB_PATH -wiki_plm_path WIKI_PLM_PATH -restore ckpt/best_checkpoint.pt
   ```

   where `-output` is a TSV file that will contain the prediction results, and `-dummy_test` is the output template to simplify output formatting. The `dummy-predictions.tsv` file is provided by the [official evaluation script](https://github.com/allenai/aristo-leaderboard/tree/master/propara/data/test) of AI2, and it is copied to the `data/` directory.

6. Run the evaluation script using the ground-truth labels and your predictions:

   ```bash
   python evaluator/evaluator.py -p data/prediction.tsv -a data/answers.tsv --diagnostics data/diagnostic.txt
   ```

   where `answers.tsv` contains the ground-truth labels, and `diagnostic.txt` will contain detailed scores for each instance. `answers.tsv` can be found [here](https://github.com/allenai/aristo-leaderboard/tree/master/propara/data/test), or you can use my copy in `data/`. `evaluator` directory contains the [evaluation scripts](https://github.com/allenai/aristo-leaderboard/tree/master/propara/evaluator) provided by AI2.


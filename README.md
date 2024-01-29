# rxn-models-for-polymerization

Description of the training procedure used in the paper [Predicting polymerization reactions via transfer learning using chemical language models"](https://doi.org/10.48550/arXiv.2310.11423)

## configure environment

```console
conda create -n rxn-models-for-polymerization python=3.7
conda activate rxn-models-for-polymerization
pip install "rdkit>=2023.3.2" "rxn-reaction-preprocessing>=2.0.2" "rxn_onmt_utils>=0.3.5" "rxn-onmt-models>=1.0.0" "rxn-metrics>=1.0.0"
```

## data preprocessing and preparation

Create data folders:

```console
mkdir -p ./data/{input,hta-allCombinations,m2p-allCombinations}
```

Download data from Zenodo for hta and m2p and move them to `./data/input/`.

Run the preprocessing pipeline (replace `/path/to` with the folder containing the cloned repo):

```console
# TODO: retrieve configuration file and run data pipeline
HYDRA_FULL_ERROR=1 rxn-data-pipeline --config-dir ./configs --config-name config_hta_allCombinations.yaml
HYDRA_FULL_ERROR=1 rxn-data-pipeline --config-dir ./configs --config-name config_m2p_allCombinations.yaml
```

Move train and validation data:

```console
# hta-allCombinations
cp ./data/rxn-preprocessing/hta-allCombinations.augmented.train.precursors_tokens ./data/hta-allCombinations/data.processed.train.precursors_tokens
cp ./data/rxn-preprocessing/hta-allCombinations.augmented.train.products_tokens ./data/hta-allCombinations/data.processed.train.products_tokens
cp ./data/rxn-preprocessing/hta-allCombinations.processed.validation.precursors_tokens ./data/hta-allCombinations/data.processed.validation.precursors_tokens
cp ./data/rxn-preprocessing/hta-allCombinations.processed.validation.products_tokens ./data/hta-allCombinations/data.processed.validation.products_tokens
# m2p-allCombinations
cp ./data/rxn-preprocessing/m2p-allCombinations.augmented.train.precursors_tokens ./data/m2p-allCombinations/data.processed.train.precursors_tokens
cp ./data/rxn-preprocessing/m2p-allCombinations.augmented.train.products_tokens ./data/m2p-allCombinations/data.processed.train.products_tokens
cp ./data/rxn-preprocessing/m2p-allCombinations.processed.validation.precursors_tokens ./data/m2p-allCombinations/data.processed.validation.precursors_tokens
cp ./data/rxn-preprocessing/m2p-allCombinations.processed.validation.products_tokens ./data/m2p-allCombinations/data.processed.validation.products_tokens
```

Prepare the data for training:

```console
# hta-allCombinations
rxn-onmt-preprocess --input_dir ./data/hta-allCombinations/ --output_dir ./data/hta-allCombinations/onmt-preprocessed-forward --model_task forward
rxn-onmt-preprocess --input_dir ./data/hta-allCombinations/ --output_dir ./data/hta-allCombinations/onmt-preprocessed-retro --model_task retro
# m2p-allCombinations
rxn-onmt-preprocess --input_dir ./data/m2p-allCombinations/ --output_dir ./data/m2p-allCombinations/onmt-preprocessed-forward --model_task forward
rxn-onmt-preprocess --input_dir ./data/m2p-allCombinations/ --output_dir ./data/m2p-allCombinations/onmt-preprocessed-retro --model_task retro
```

## training

Assuming base models are stored in a `./models` folder, run the training pipeline:

```console
# hta-allCombinations
rxn-onmt-finetune --model_output_dir ./hta-allCombinations_forward --preprocess_dir ./data/hta-allCombinations/onmt-preprocessed-forward  --train_from ./models/forward-base-model.pt --train_num_steps 20000 --learning_rate 0.06 --batch_size 512  # NOTE: optionally to control save frequency: --save_checkpoint_steps 500 --report_every 100
rxn-onmt-finetune --model_output_dir ./hta-allCombinations_retro --preprocess_dir ./data/hta-allCombinations/onmt-preprocessed-retro  --train_from ./models/backward-base-model.pt --train_num_steps 20000 --learning_rate 0.6 --batch_size 512  # NOTE: optionally to control save frequency: --save_checkpoint_steps 500 --report_every 100
# m2p-allCombinations
rxn-onmt-finetune --model_output_dir ./m2p-allCombinations_forward --preprocess_dir ./data/m2p-allCombinations/onmt-preprocessed-forward  --train_from ./models/forward-base-model.pt --train_num_steps 20000 --learning_rate 0.06 --batch_size 512  # NOTE: optionally to control save frequency: --save_checkpoint_steps 500 --report_every 100
rxn-onmt-finetune --model_output_dir ./m2p-allCombinations_retro --preprocess_dir ./data/m2p-allCombinations/onmt-preprocessed-retro  --train_from ./models/backward-base-model.pt --train_num_steps 20000 --learning_rate 0.6 --batch_size 512  # NOTE: optionally to control save frequency: --save_checkpoint_steps 500 --report_every 100
```

*NOTE:* we do not redistribute the model weights used in the publication for licensing reasons. To generate both the retro and the forward base model one can use [`rxn-onmt-models`](https://github.com/rxn4chemistry/rxn-onmt-models/tree/main) and a reaction dataset such as the [Chemical reactions from US patents (1976-Sep2016)](https://figshare.com/articles/dataset/Chemical_reactions_from_US_patents_1976-Sep2016_/5104873).

## evaluation

Run the models evaluation:

```console
export steps=20000  # NOTE: or specify the number of steps for the desired checkpoint
# hta-allCombinations
rxn-prepare-retro-metrics --precursors_file ./data/rxn-preprocessing/hta-allCombinations.processed.test.precursors_tokens --products_file ./data/rxn-preprocessing/hta-allCombinations.processed.test.products_tokens --forward_model ./hta-allCombinations_forward/model_step_${step}.pt --retro_model ./hta-allCombinations_retro/model_step_${step}.pt --output_dir ./hta-allCombinations_retro_evaluation_${step} --n_best 10
rxn-prepare-forward-metrics --precursors_file ./data/rxn-preprocessing/hta-allCombinations.processed.test.precursors_tokens --products_file ./data/rxn-preprocessing/hta-allCombinations.processed.test.products_tokens --forward_model ./hta-allCombinations_forward/model_step_${step}.pt --output_dir ./hta-allCombinations_forward_evaluation_${step} --n_best 10
# m2p-allCombinations
rxn-prepare-retro-metrics --precursors_file ./data/rxn-preprocessing/m2p-allCombinations.processed.test.precursors_tokens --products_file ./data/rxn-preprocessing/m2p-allCombinations.processed.test.products_tokens --forward_model ./m2p-allCombinations_forward/model_step_${step}.pt --retro_model ./m2p-allCombinations_retro/model_step_${step}.pt --output_dir ./m2p-allCombinations_retro_evaluation_${step} --n_best 10
rxn-prepare-forward-metrics --precursors_file ./data/rxn-preprocessing/m2p-allCombinations.processed.test.precursors_tokens --products_file ./data/rxn-preprocessing/m2p-allCombinations.processed.test.products_tokens --forward_model ./m2p-allCombinations_forward/model_step_${step}.pt --output_dir ./m2p-allCombinations_forward_evaluation_${step} --n_best 10
```

Merge everything in a single .csv file:

```console
rxn-parse-metrics-into-csv --csv ./data/metrics_allCombinations.csv ./{hta,m2p}-allCombinations_{retro,forward}_*
```

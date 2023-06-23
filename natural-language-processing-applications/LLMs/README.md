# nlp-final

Training code (run.py) taken from https://github.com/gregdurrett/fp-dataset-artifacts and modified further

Forgetting examples (Yaghoobzadeh et al (2021)) implemented in (trainer.py, helpers.py)

Counterfactually-augmented data (Kaushik et al (2020)) taken from https://github.com/acmi-lab/counterfactually-augmented-data and reformated to match SNLI dataset on huggingface https://huggingface.co/datasets/snli/viewer/plain_text/test (via format_data.py)

python3 run.py --do_train --task nli --dataset snli --output_dir ./trained_model/
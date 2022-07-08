
wget https://raw.githubusercontent.com/huggingface/transformers/main/examples/pytorch/language-modeling/run_plm.py
wget https://raw.githubusercontent.com/huggingface/transformers/main/examples/research_projects/mlm_wwm/run_mlm_wwm.py
pip install git+https://github.com/huggingface/transformers datasets
python make_pretrain_dataset.py

python run_mlm_wwm.py --model_name_or_path=roberta-base \
                  --use_auth_token \
                  --gradient_accumulation_steps=4 \
                  --push_to_hub_model_id=nbme-roberta-base \
                  --train_file=data/train.txt \
                  --validation_file=data/eval.txt \
                  --save_strategy=epoch \
                  --output_dir=models \
                  --do_train \
                  --overwrite_output_dir \
                  --pad_to_max_length \
                  --load_best_model_at_end \
                  --metric_for_best_model=eval_loss \
                  --evaluation_strategy=epoch \
                  --push_to_hub

python run_mlm_wwm.py --model_name_or_path=yikuan8/Clinical-Longformer \
                  --use_auth_token \
                  --gradient_accumulation_steps=4 \
                  --push_to_hub_model_id=nbme-clinical-longformer \
                  --train_file=data/train.txt \
                  --validation_file=data/eval.txt \
                  --save_strategy=epoch \
                  --output_dir=models \
                  --do_train \
                  --overwrite_output_dir \
                  --pad_to_max_length \
                  --load_best_model_at_end \
                  --metric_for_best_model=eval_loss \
                  --evaluation_strategy=epoch \
                  --push_to_hub

python run_mlm_wwm.py --model_name_or_path=emilyalsentzer/Bio_ClinicalBERT\
                  --use_auth_token \
                  --gradient_accumulation_steps=4 \
                  --push_to_hub_model_id=nbme-bio-clinical-bert \
                  --train_file=data/train.txt \
                  --validation_file=data/eval.txt \
                  --save_strategy=epoch \
                  --output_dir=models \
                  --do_train \
                  --overwrite_output_dir \
                  --pad_to_max_length \
                  --load_best_model_at_end \
                  --metric_for_best_model=eval_loss \
                  --evaluation_strategy=epoch \
                  --push_to_hub

wget https://raw.githubusercontent.com/huggingface/transformers/main/examples/pytorch/language-modeling/run_plm.py
wget https://raw.githubusercontent.com/huggingface/transformers/main/examples/research_projects/mlm_wwm/run_mlm_wwm.py
wget https://raw.githubusercontent.com/huggingface/transformers/main/examples/pytorch/language-modeling/run_clm.py
pip install git+https://github.com/huggingface/transformers datasets
python make_pretrain_dataset.py

python run_mlm_wwm.py --model_name_or_path=microsoft/deberta-large \
                  --use_auth_token \
                  --gradient_accumulation_steps=4 \
                  --push_to_hub_model_id=nbme-deberta-large \
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
                  --fp16 \
                  --push_to_hub

python run_mlm_wwm.py --model_name_or_path=roberta-large \
                  --use_auth_token \
                  --gradient_accumulation_steps=4 \
                  --push_to_hub_model_id=nbme-roberta-large \
                  --train_file=data/train.txt \
                  --validation_file=data/eval.txt \
                  --save_strategy=epoch \
                  --output_dir=models \
                  --do_train \
                  --overwrite_output_dir \
                  --pad_to_max_length \
                  --load_best_model_at_end \
                  --metric_for_best_model=eval_loss \
                  --evaluation_strategy=epoch  \
                  --push_to_hub

python run_mlm_wwm.py --model_name_or_path=microsoft/deberta-v3-large \
                  --use_auth_token \
                  --gradient_accumulation_steps=4 \
                  --push_to_hub_model_id=nbme-deberta-v3-large \
                  --train_file=data/train.txt \
                  --validation_file=data/eval.txt \
                  --save_strategy=epoch \
                  --output_dir=models \
                  --do_train \
                  --overwrite_output_dir \
                  --pad_to_max_length \
                  --load_best_model_at_end \
                  --metric_for_best_model=eval_loss \
                  --evaluation_strategy=epoch  \
                  --push_to_hub

python run_mlm_wwm.py --model_name_or_path=google/electra-large-discriminator \
                  --use_auth_token \
                  --gradient_accumulation_steps=4 \
                  --push_to_hub_model_id=nbme-electra-large-discriminator \
                  --train_file=data/train.txt \
                  --validation_file=data/eval.txt \
                  --save_strategy=epoch \
                  --output_dir=models \
                  --do_train \
                  --overwrite_output_dir \
                  --pad_to_max_length \
                  --load_best_model_at_end \
                  --metric_for_best_model=eval_loss \
                  --evaluation_strategy=epoch  \
                  --push_to_hub

python run_plm.py --model_name_or_path=xlnet-large-cased \
                  --use_auth_token \
                  --pad_to_max_length \
                  --line_by_line \
                  --gradient_accumulation_steps=4 \
                  --push_to_hub_model_id=nbme-xlnet-large-cased \
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

python run_mlm_wwm.py --model_name_or_path=microsoft/deberta-v2-xlarge \
                  --use_auth_token \
                  --gradient_accumulation_steps=8 \
                  --per_device_train_batch_size=4 \
                  --push_to_hub_model_id=nbme-deberta-v2-xlarge \
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
                  --fp16 \
                  --push_to_hub


python run_clm.py --model_name_or_path=google/electra-large-generator \
                  --use_auth_token \
                  --gradient_accumulation_steps=4 \
                  --push_to_hub_model_id=nbme-electra-large-generator \
                  --train_file=data/train.txt \
                  --validation_file=data/eval.txt \
                  --save_strategy=epoch \
                  --output_dir=models \
                  --do_train \
                  --overwrite_output_dir \
                  --pad_to_max_length \
                  --load_best_model_at_end \
                  --metric_for_best_model=eval_loss \
                  --evaluation_strategy=epoch  \
                  --push_to_hub

python run_clm.py --model_name_or_path=gpt2 \
                  --use_auth_token \
                  --gradient_accumulation_steps=8 \
                  --per_device_train_batch_size=4 \
                  --push_to_hub_model_id=nbme-gpt2 \
                  --train_file=data/train.txt \
                  --validation_file=data/eval.txt \
                  --save_strategy=epoch \
                  --output_dir=models \
                  --do_train \
                  --overwrite_output_dir \
                  --load_best_model_at_end \
                  --metric_for_best_model=eval_loss \
                  --evaluation_strategy=epoch  \
                  --push_to_hub

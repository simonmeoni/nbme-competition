python pretrain_model.py --model_name="distilroberta-base" \
                         --huggingface_repository="smeoni/nbme-distilroberta-base"
python run.py -m current_fold=0,1,2,3,4 \
                 model.model.model='smeoni/nbme-distilroberta-base' \
                 experiment_name=pretraining-nbme \
                 model.loss._target_=torch.nn.BCEWithLogitsLoss

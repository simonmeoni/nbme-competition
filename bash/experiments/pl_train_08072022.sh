pip install git+https://github.com/huggingface/transformers.git
python run.py -m model.loss._target_=torch.nn.BCEWithLogitsLoss \
                 model.model.model="smeoni/nbme-deberta-large,smeoni/nbme-deberta-v3-large" \
                 current_fold=0 \
                 datamodule.pl_train_mode=True \
                 datamodule.batch_size=8 \
                 trainer.max_epochs=5 \
                 +trainer.accumulate_grad_batches=4 \
                 experiment_name="pl_train"
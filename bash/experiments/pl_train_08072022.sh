pip install git+https://github.com/huggingface/transformers.git
python run.py -m model.loss._target_=torch.nn.BCEWithLogitsLoss \
                 model.model.model="smeoni/nbme-deberta-large,smeoni/nbme-deberta-V3-large" \
                 current_fold=0 \
                 datamodule.pl_mode=0.2,0.6,1 \
                 datamodule.batch_size=4 \
                 trainer.max_epochs=6 \
                 +trainer.accumulate_grad_batches=16 \
                 experiment_name="pl_train"

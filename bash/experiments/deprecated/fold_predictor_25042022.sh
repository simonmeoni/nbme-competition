pip install git+https://github.com/huggingface/transformers.git

models="deberta-large,microsoft/deberta-v3-large"

python run.py -m model.loss._target_=torch.nn.BCEWithLogitsLoss \
                 model.model.model=$models \
                 current_fold=0 \
                 +trainer.precision=16 \
                 datamodule.batch_size=8 \
                 trainer.max_epochs=5 \
                 +trainer.accumulate_grad_batches=4 \
                 experiment_name="big_train"

models="microsoft/deberta-v2-xlarge"
python run.py -m model.loss._target_=torch.nn.BCEWithLogitsLoss \
                 model.model.model=$models \
                 current_fold=0 \
                 +trainer.precision=16 \
                 datamodule.batch_size=4 \
                 trainer.max_epochs=5 \
                 +trainer.accumulate_grad_batches=8 \
                 experiment_name="big_train"

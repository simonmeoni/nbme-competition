pip install git+https://github.com/huggingface/transformers.git

models="microsoft/deberta-v3-large,smeoni/nbme-deberta-v3-large"

python run.py -m model.loss._target_=torch.nn.BCEWithLogitsLoss \
                 model.model.model=$models \
                 model.optimizer.lr=3e-5 \
                 current_fold=0,1,2,3,4 \
                 +trainer.precision=16 \
                 datamodule.batch_size=8 \
                 trainer.max_epochs=10 \
                 +trainer.accumulate_grad_batches=4 \
                 experiment_name="big_train_1"

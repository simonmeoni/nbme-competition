pip install git+https://github.com/huggingface/transformers.git

models="microsoft/deberta-v2-xlarge,smeoni/nbme-deberta-v2-xlarge"

python run.py -m model.loss._target_=torch.nn.BCEWithLogitsLoss \
                 model.model.model=$models \
                 current_fold=0,1,2,3,4 \
                 +trainer.precision=16 \
                 model.optimizer.lr=2e-5 \
                 datamodule.batch_size=4 \
                 trainer.max_epochs=10 \
                 +trainer.accumulate_grad_batches=8 \
                 experiment_name="big_train"

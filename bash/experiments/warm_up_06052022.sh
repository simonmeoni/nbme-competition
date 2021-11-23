pip install git+https://github.com/huggingface/transformers.git
python run.py    model.loss._target_=torch.nn.BCEWithLogitsLoss \
                 model.model.model="smeoni/nbme-deberta-large" \
                 model.optimizer.lr=3e-5 \
                 current_fold=0 \
                 +trainer.precision=16 \
                 datamodule.batch_size=8 \
                 model.scheduler._target_=transformers.get_linear_schedule_with_warmup \
                 ~model.scheduler.patience \
                 model.scheduler.num_warmup_steps=100 \
                 model.scheduler.num_training_steps=1007 \
                 model.watcher.interval="step" \
                 ~model.watcher.monitor \
                 trainer.max_epochs=6 \
                 +trainer.accumulate_grad_batches=4 \
                 experiment_name="one_fold_warm_up"

python run.py    model.loss._target_=torch.nn.BCEWithLogitsLoss \
                 model.model.model="smeoni/nbme-deberta-large" \
                 model.optimizer.lr=3e-5 \
                 current_fold=0 \
                 +trainer.precision=16 \
                 datamodule.batch_size=8 \
                 trainer.max_epochs=6 \
                 +trainer.accumulate_grad_batches=4 \
                 experiment_name="one_fold_warm_up"

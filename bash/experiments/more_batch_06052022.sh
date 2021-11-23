pip install git+https://github.com/huggingface/transformers.git

python run.py model.loss._target_=torch.nn.BCEWithLogitsLoss \
                 model.model.model=smeoni/nbme-deberta-large \
                 model.optimizer.lr=3e-5 \
                 current_fold=0 \
                 datamodule.k_fold=4 \
                 +trainer.gpus=[0] \
                 +trainer.precision=16 \
                 datamodule.batch_size=8 \
                 trainer.max_epochs=6 \
                 +trainer.accumulate_grad_batches=8 \
                 experiment_name="more_batch"

python run.py model.loss._target_=torch.nn.BCEWithLogitsLoss \
                 model.model.model=smeoni/nbme-deberta-large \
                 model.optimizer.lr=3e-5 \
                 current_fold=1 \
                 datamodule.k_fold=4 \
                 +trainer.gpus=[1] \
                 +trainer.precision=16 \
                 datamodule.batch_size=8 \
                 trainer.max_epochs=6 \
                 +trainer.accumulate_grad_batches=8 \
                 experiment_name="more_batch"

python run.py model.loss._target_=torch.nn.BCEWithLogitsLoss \
                 model.model.model=smeoni/nbme-deberta-large \
                 model.optimizer.lr=3e-5 \
                 current_fold=2 \
                 datamodule.k_fold=4 \
                 +trainer.gpus=[2] \
                 +trainer.precision=16 \
                 datamodule.batch_size=8 \
                 trainer.max_epochs=6 \
                 +trainer.accumulate_grad_batches=8 \
                 experiment_name="more_batch"

python run.py model.loss._target_=torch.nn.BCEWithLogitsLoss \
                 model.model.model=smeoni/nbme-deberta-large \
                 model.optimizer.lr=3e-5 \
                 current_fold=3 \
                 datamodule.k_fold=4 \
                 +trainer.gpus=[3] \
                 +trainer.precision=16 \
                 datamodule.batch_size=8 \
                 trainer.max_epochs=6 \
                 +trainer.accumulate_grad_batches=8 \
                 experiment_name="more_batch"

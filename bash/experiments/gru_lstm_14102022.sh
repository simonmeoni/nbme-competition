#!/usr/bin/env bash
python run.py -m model.model.model="smeoni/nbme-deberta-large,deberta-large,deepset/deberta-v3-large-squad2" \
                 model="gru,lstm" \
                 current_fold=0 \
                 model.optimizer.lr=4e-5 \
                 model.optimizer.weight_decay=1e-6 \
                 model.scheduler._target_=transformers.get_linear_schedule_with_warmup \
                 ~model.scheduler.patience \
                 +model.scheduler.num_warmup_steps=600 \
                 +model.scheduler.num_training_steps=9627 \
                 model.watcher.interval="step" \
                 ~model.watcher.monitor \
                 datamodule.pl_mode=1 \
                 datamodule.batch_size=4 \
                 trainer.max_epochs=6 \
                 +trainer.accumulate_grad_batches=16 \
                 experiment_name="heads_and_seeds" \
                 seed=12345,42,10 \
                 ~logger.wandb.name \
                 logger.wandb.group="heads_and_seeds"

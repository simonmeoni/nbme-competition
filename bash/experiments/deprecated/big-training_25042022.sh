pip install git+https://github.com/huggingface/transformers.git

models="roberta-base,nbme-roberta-base,emilyalsentzer/Bio_ClinicalBERT,smeoni/nbme-Bio_ClinicalBERT"
python run.py -m model.loss._target_=torch.nn.BCEWithLogitsLoss \
                 model.model.model=$models \
                 current_fold=0,1,2,3,4 \
                 trainer.max_epochs=10 \
                 experiment_name="big_train"

models="deberta-large,microsoft/deberta-v3-large,smeoni/nbme-deberta-v3-large,\
smeoni/nbme-deberta-large,xlnet-large-cased,smeoni/nbme-xlnet-large-cased,\
google/electra-large-discriminator,google/electra-large-generator,\
smeoni/nbme-electra-large-generator,roberta-large,nbme-roberta-large,yikuan8/Clinical-Longformer,\
smeoni/nbme-clinical-longformer"

python run.py -m model.loss._target_=torch.nn.BCEWithLogitsLoss \
                 model.model.model=$models \
                 current_fold=0,1,2,3,4 \
                 +trainer.precision=16 \
                 datamodule.batch_size=8 \
                 trainer.max_epochs=10 \
                 +trainer.accumulate_grad_batches=4 \
                 experiment_name="big_train"

models="microsoft/deberta-v2-xlarge,smeoni/nbme-deberta-v2-xlarge"
python run.py -m model.loss._target_=torch.nn.BCEWithLogitsLoss \
                 model.model.model=$models \
                 current_fold=0,1,2,3,4 \
                 +trainer.precision=16 \
                 datamodule.batch_size=4 \
                 trainer.max_epochs=10 \
                 +trainer.accumulate_grad_batches=8 \
                 experiment_name="big_train"

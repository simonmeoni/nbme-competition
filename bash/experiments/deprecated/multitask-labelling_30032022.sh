architectures="src.models.architectures.linearclassifier.ThreeHeadsLinearClassifier,\
src.models.architectures.linearclassifier.TwoHeadsLinearClassifier"
losses="torch.nn.BCEWithLogitsLoss,\
src.utils.dice_loss.DiceLoss,\
src.utils.dice_loss.BCEDiceLoss"
experiment_name="multitask-labelling"

echo $losses
echo $architectures
echo $experiment_name
num_classes=1,2,3
sleep 10s
python run.py -m ++trainer.fast_dev_run=True \
                 datamodule.batch_size=4 logger=csv callbacks=none \
                 model.loss._target_=$losses \
                 model.model._target_=$architectures \
                 experiment_name=$experiment_name

python run.py -m ++trainer.fast_dev_run=True \
                 datamodule.batch_size=4 logger=csv callbacks=none \
                 model.loss._target_=$losses \
                 model.model._target_=src.models.architectures.linearclassifier.LinearClassifier \
                 +model.model.num_classes=$num_classes \
                 experiment_name=$experiment_name
echo "******************************"
echo "*    Test are finished       *"
echo "******************************"
sleep 10s
python run.py -m model.loss._target_=$losses \
                 model.model._target_=$architectures \
                 current_fold=0,1,2,3,4 \
                 experiment_name=$experiment_name

python run.py -m model.loss._target_=$losses \
                 model.model._target_=src.models.architectures.linearclassifier.LinearClassifier \
                 +model.model.num_classes=$num_classes \
                 current_fold=0,1,2,3,4 \
                 experiment_name=$experiment_name

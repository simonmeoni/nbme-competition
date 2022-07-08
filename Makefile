include .env
export

get_dataset:
	kaggle competitions download -c nbme-score-clinical-patient-notes -p data
	unzip -o data/nbme-score-clinical-patient-notes.zip -d data/nbme-score-clinical-patient-notes
	kaggle datasets download -d simonmeoni/nbme-merge-dataset -p data/nbme-merge-dataset --unzip
	kaggle datasets download -d simonmeoni/pl-train-nbme-dataset -p data/pl-train-nbme-dataset --unzip
	rm data/*.zip

experiment:
	 nohup python run.py -m experiment=$(experiment) ++trainer.gpus=1 &
upload-checkpoints:
	python bin/upload_checkpoints.py \
        --wandb_project simonmeoni/nbme-competition \
        --kaggle_dataset ($kaggle_dataset)\
        --checkpoints_path models/checkpoints \
        --wandb_groups $(wandb_groups)
login:
	bash/utils/login.sh

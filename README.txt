Running Commands (with default experiment settings):
	
	For sequence classification tasks,  
		1. text classification: 
			python start.py --cfg MLDoc_bert.cfg
		2. sentiment analysis: 
			python start.py --cfg SC2_bert.cfg
		
	For information classification task,
		1. relation extraction: 
			python example/train_supervised_bert.py
		
	For question answering task,
		1. question answer: 
			python run_bipar.py

init:
	conda install -c conda-forge matplotlib
	conda install --yes --file requirements.txt
show:
	python main.py

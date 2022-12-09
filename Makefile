fast: pytest_fast mypy flake
all: coverage flake mypy doctest scripts
flake:
	flake8 --max-line-length 99 --ignore E203
mypy:
	mypy src/kalman_experiments/*.py
	mypy src/kalman_experiments/kalman/*.py
coverage:
	coverage run -m pytest
	coverage report
doctests_coverage:
	coverage run metapipe/file_processors.py
	coverage report
pytest_fast:
	pytest -m "not slow"
doctest:
	python src/kalman_experiments/model_selection.py
scripts:
	python ./notebooks/mk_psd_study.sync.py
	python ./notebooks/sspe.sync.py
	python ./notebooks/spectral_fit.sync.py
	python ./notebooks/roc_bci.sync.py

.PHONY: figures clean

figures:
	cd scripts/04_figures && python performance_figures.py && python biomarker_figures.py

clean:
	rm -rf scripts/04_figures/output scripts/04_figures/__pycache__

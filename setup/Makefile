VENV=$(CURDIR)/venv

ALL: notebook

$(VENV):
	python3 -m venv $(VENV)

$(VENV)/bin/jupyter-notebook: $(VENV)
	$(VENV)/bin/pip install -r requirements.txt

notebook: $(VENV)/bin/jupyter-notebook
	$(VENV)/bin/jupyter-notebook

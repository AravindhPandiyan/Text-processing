.PHONY: tests docs

deps: 
	@echo "Initializing Git..."
	git init
	
	@echo "Installing dependencies..."
	poetry install --no-root
	poetry run pre-commit install

gen_req:
	@echo Generating requirements.txt file...
	poetry export -f requirements.txt --without-hashes --output requirements.txt

docs_host:
	@echo View API documentation...
	pdoc src --http localhost:8080

docs_save:
	@echo Save documentation to docs...
	pdoc --pdf src > docs/src.md

docs_pdf:
	@echo Generating PDF from docs...
	pandoc docs/src.md -o docs/src.pdf

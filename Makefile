.PHONY: install test test-js viewer-js docs docs-serve clean

install: ## Install bifrost with its dev and docs extras
	pip install -e ".[dev,docs]"

test: ## Run the Python tests
	pytest tests

test-js: ## Run the viewer's JS tests and typecheck
	cd js && npm install && npm test && npm run typecheck

viewer-js: ## Rebuild the committed viewer bundle from js/ (output: bifrost/viewer/static)
	cd js && npm install && npm run build && npm run typecheck

docs: ## Build the docs site (strict)
	zensical build --strict -f docs/zensical.yml

docs-serve: ## Serve the docs locally
	zensical serve -f docs/zensical.yml -a localhost:8080

clean:
	rm -rf docs/_build build dist

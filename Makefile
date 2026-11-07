.PHONY: install test test-js test-docs test-hardware viewer-js demo demo-notebook docs docs-serve clean

install: ## Install bifrost with its dev and docs extras
	pip install -e ".[dev,docs]"

test: ## Run the Python tests
	pytest tests

test-js: ## Run the viewer's JS tests and typecheck
	cd js && npm install && npm test && npm run typecheck

test-docs: ## Check the docs: snippets, links, install commands, examples, strict build, demo in a browser
	pytest tests/test_docs.py tests/test_demo_browser.py tests/test_viewer_export.py

test-hardware: ## Run the tutorial's full simulations (needs a GPU)
	BIFROST_HARDWARE=1 pytest tests/test_docs.py -k hardware

demo: viewer-js ## Regenerate the live demo's data and poster (needs a GPU; commit docs/demo afterwards)
	python scripts/build_demo.py
	python scripts/capture_demo_poster.py

viewer-js: ## Rebuild the committed viewer bundle from js/ (output: bifrost/viewer/static)
	cd js && npm install && npm run build && npm run typecheck

demo-notebook: ## Export the Widget Demo notebook (docs/notebooks/widget_demo.py) to docs/demo/notebook
	marimo export html-wasm docs/notebooks/widget_demo.py -o docs/demo/notebook --mode run --no-show-code --no-sandbox -f
	rm -f docs/demo/notebook/CLAUDE.md  # marimo's editor-assistant prompt: no use in a run-mode export

docs: demo-notebook ## Build the docs site (strict)
	zensical build --strict -f docs/zensical.yml

docs-serve: demo-notebook ## Serve the docs locally
	zensical serve -f docs/zensical.yml -a localhost:8080

clean:
	rm -rf docs/_build docs/demo/notebook build dist

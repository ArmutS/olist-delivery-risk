.PHONY: install download-data audit run lint test check

install:
	python -m pip install -e '.[dev]'

download-data:
	mkdir -p data/raw
	kaggle datasets download -d olistbr/brazilian-ecommerce -p data/raw --unzip

audit:
	olist-risk audit

run:
	olist-risk run

lint:
	ruff check src tests

test:
	pytest

check: lint test

.PHONY: data triplets eval clean pipeline

data:
	uv run python src/data/generate.py --config configs/dataset.yaml --output-dir data/

triplets:
	uv run python src/data/triplets.py --config configs/dataset.yaml --profiles data/processed/triplet_source.parquet --output-dir data/triplets/

eval:
	uv run python src/data/eval_set.py --config configs/dataset.yaml --eval-profiles data/eval/eval_profiles.parquet --output-dir data/eval/

clean:
	rm -f data/raw/*.parquet data/processed/*.parquet data/processed/*.json data/triplets/*.parquet data/eval/*.parquet data/eval/*.json

pipeline: data triplets eval

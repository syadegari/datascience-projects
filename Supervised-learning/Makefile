FEATURE_FILES := data/features_1d_*.parquet
TARGET_FILE   := data/target.parquet
MATRIX_FILE   := data/model_matrix.parquet
SPLIT_DIR     := data/splits

all: data_pipeline target_generation merge_features_labels apply_splits


data_pipeline:
	python data_pipeline.py
	@echo

target_generation: 
	python target_generation.py $(FEATURE_FILES) $(TARGET_FILE) \
		--horizon 5 \
		--pct 0.03 \
		--direction up
	@echo

merge_features_labels: 
	python merge_features_labels.py \
		--features $(FEATURE_FILES) \
		--labels   $(TARGET_FILE) \
		--output   $(MATRIX_FILE)
	@echo

apply_splits: 
	python apply_splits.py \
		--model-matrix $(MATRIX_FILE) \
		--split-plan   configs/split_plan.yaml \
		--out-dir      $(SPLIT_DIR)
	@echo

clean:
	rm -f $(TARGET_FILE) $(MATRIX_FILE)
	rm -rf $(SPLIT_DIR)

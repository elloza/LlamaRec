rm -rf ./data/*
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train_retriever.py
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train_ranker.py
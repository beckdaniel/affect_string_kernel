OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt ridge warp ../results &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt svr warp ../results &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt rbf warp ../results &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt mat32 warp ../results &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt mat52 warp ../results &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt ratquad warp ../results &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt linear warp ../results &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt mlp warp ../results &

OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt ridge warp ../results --ard &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt svr warp ../results --ard &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt rbf warp ../results --ard &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt mat32 warp ../results --ard &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt mat52 warp ../results --ard &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt ratquad warp ../results --ard &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt linear warp ../results --ard &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt mlp warp ../results --ard &

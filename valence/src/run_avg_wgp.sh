OMP_NUM_THREADS=1 python avg_model.py ../../data/instances.txt ../../data/affectivetext_test.valence.gold ../../embs/glove.6B.100d.txt rbf warp ../results &
OMP_NUM_THREADS=1 python avg_model.py ../../data/instances.txt ../../data/affectivetext_test.valence.gold ../../embs/glove.6B.100d.txt mat32 warp ../results &
OMP_NUM_THREADS=1 python avg_model.py ../../data/instances.txt ../../data/affectivetext_test.valence.gold ../../embs/glove.6B.100d.txt mat52 warp ../results &
OMP_NUM_THREADS=1 python avg_model.py ../../data/instances.txt ../../data/affectivetext_test.valence.gold ../../embs/glove.6B.100d.txt ratquad warp ../results &
OMP_NUM_THREADS=1 python avg_model.py ../../data/instances.txt ../../data/affectivetext_test.valence.gold ../../embs/glove.6B.100d.txt linear warp ../results &
OMP_NUM_THREADS=1 python avg_model.py ../../data/instances.txt ../../data/affectivetext_test.valence.gold ../../embs/glove.6B.100d.txt mlp warp ../results &

OMP_NUM_THREADS=1 python avg_model.py ../../data/instances.txt ../../data/affectivetext_test.valence.gold ../../embs/glove.6B.100d.txt rbf warp ../results --ard &
OMP_NUM_THREADS=1 python avg_model.py ../../data/instances.txt ../../data/affectivetext_test.valence.gold ../../embs/glove.6B.100d.txt mat32 warp ../results --ard &
OMP_NUM_THREADS=1 python avg_model.py ../../data/instances.txt ../../data/affectivetext_test.valence.gold ../../embs/glove.6B.100d.txt mat52 warp ../results --ard &
OMP_NUM_THREADS=1 python avg_model.py ../../data/instances.txt ../../data/affectivetext_test.valence.gold ../../embs/glove.6B.100d.txt ratquad warp ../results --ard &
OMP_NUM_THREADS=1 python avg_model.py ../../data/instances.txt ../../data/affectivetext_test.valence.gold ../../embs/glove.6B.100d.txt linear warp ../results --ard &
OMP_NUM_THREADS=1 python avg_model.py ../../data/instances.txt ../../data/affectivetext_test.valence.gold ../../embs/glove.6B.100d.txt mlp warp ../results --ard &

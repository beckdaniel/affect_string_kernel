OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt ridge scale ../results &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt svr scale ../results &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt rbf scale ../results &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt mat32 scale ../results &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt mat52 scale ../results &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt ratquad scale ../results &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt linear scale ../results &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt mlp scale ../results &

OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt ridge scale ../results --ard &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt svr scale ../results --ard &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt rbf scale ../results --ard &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt mat32 scale ../results --ard &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt mat52 scale ../results --ard &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt ratquad scale ../results --ard &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt linear scale ../results --ard &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt mlp scale ../results --ard &

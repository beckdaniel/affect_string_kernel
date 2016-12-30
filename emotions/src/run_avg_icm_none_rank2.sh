OMP_NUM_THREADS=1 python avg_model_icm.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt ridge none ../results --rank 2 &
OMP_NUM_THREADS=1 python avg_model_icm.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt svr none ../results --rank 2 &
#OMP_NUM_THREADS=1 python avg_model_icm.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt rbf none ../results &
OMP_NUM_THREADS=1 python avg_model_icm.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt mat32 none ../results --rank 2 &
OMP_NUM_THREADS=1 python avg_model_icm.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt mat52 none ../results --rank 2 &
OMP_NUM_THREADS=1 python avg_model_icm.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt ratquad none ../results --rank 2 &
OMP_NUM_THREADS=1 python avg_model_icm.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt linear none ../results --rank 2 &
OMP_NUM_THREADS=1 python avg_model_icm.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt mlp none ../results --rank 2 &

OMP_NUM_THREADS=1 python avg_model_icm.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt ridge none ../results --ard --rank 2 &
OMP_NUM_THREADS=1 python avg_model_icm.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt svr none ../results --ard --rank 2 &
#OMP_NUM_THREADS=1 python avg_model_icm.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt rbf none ../results --ard --rank 2 &
OMP_NUM_THREADS=1 python avg_model_icm.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt mat32 none ../results --ard --rank 2 &
OMP_NUM_THREADS=1 python avg_model_icm.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt mat52 none ../results --ard --rank 2 &
OMP_NUM_THREADS=1 python avg_model_icm.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt ratquad none ../results --ard --rank 2 &
OMP_NUM_THREADS=1 python avg_model_icm.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt linear none ../results --ard --rank 2 &
OMP_NUM_THREADS=1 python avg_model_icm.py ../../data/instances.txt ../../data/affectivetext_test.emotions.gold ../../embs/glove.6B.100d.txt mlp none ../results --ard --rank 2 &

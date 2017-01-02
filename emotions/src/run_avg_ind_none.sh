OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt ridge none ../results &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt svr none ../results &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt rbf none ../results &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt mat32 none ../results &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt mat52 none ../results &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt ratquad none ../results &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt linear none ../results &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt mlp none ../results &

OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt rbf none ../results --bias &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt mat32 none ../results --bias &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt mat52 none ../results --bias &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt ratquad none ../results --bias &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt linear none ../results --bias &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt mlp none ../results --bias &

#OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt ridge none ../results --ard &
#OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt svr none ../results --ard &
#OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt rbf none ../results --ard &
#OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt mat32 none ../results --ard &
#OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt mat52 none ../results --ard &
#OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt ratquad none ../results --ard &
#OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt linear none ../results --ard &
#OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt mlp none ../results --ard &

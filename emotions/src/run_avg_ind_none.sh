OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt ridge none ../results > nohups/ridge &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt svr none ../results > nohups/svr &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt rbf none ../results > nohups/rbf &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt mat32 none ../results > nohups/mat32 &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt mat52 none ../results > nohups/mat52 &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt ratquad none ../results > nohups/ratquad &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt linear none ../results > nohups/linear &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt mlp none ../results > nohups/mlp &

OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt rbf none ../results --bias > nohups/rbf_bias &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt mat32 none ../results --bias > nohups/mat32_bias &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt mat52 none ../results --bias > nohups/mat52_bias &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt ratquad none ../results --bias > nohups/ratquad_bias &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt linear none ../results --bias > nohups/linear_bias &
OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt mlp none ../results --bias > nohups/mlp_bias &

#OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt ridge none ../results --ard &
#OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt svr none ../results --ard &
#OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt rbf none ../results --ard &
#OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt mat32 none ../results --ard &
#OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt mat52 none ../results --ard &
#OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt ratquad none ../results --ard &
#OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt linear none ../results --ard &
#OMP_NUM_THREADS=1 python avg_model_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt mlp none ../results --ard &

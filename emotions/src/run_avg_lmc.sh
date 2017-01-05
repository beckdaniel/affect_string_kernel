OMP_NUM_THREADS=1 nohup python avg_model_lmc.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt mat32 none ../results --rank 1 > lmc_nohups/mat32_1.nohup &
OMP_NUM_THREADS=1 nohup python avg_model_lmc.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt mat32 none ../results --rank 2 > lmc_nohups/mat32_2.nohup &
OMP_NUM_THREADS=1 nohup python avg_model_lmc.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt mat32 none ../results --rank 3 > lmc_nohups/mat32_3.nohup &
OMP_NUM_THREADS=1 nohup python avg_model_lmc.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt mat32 none ../results --rank 4 > lmc_nohups/mat32_4.nohup &
OMP_NUM_THREADS=1 nohup python avg_model_lmc.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt mat32 none ../results --rank 5 > lmc_nohups/mat32_5.nohup &

OMP_NUM_THREADS=1 nohup python avg_model_lmc.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt mat52 none ../results --rank 1 > lmc_nohups/mat52_1.nohup &
OMP_NUM_THREADS=1 nohup python avg_model_lmc.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt mat52 none ../results --rank 2 > lmc_nohups/mat52_2.nohup &
OMP_NUM_THREADS=1 nohup python avg_model_lmc.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt mat52 none ../results --rank 3 > lmc_nohups/mat52_3.nohup &
OMP_NUM_THREADS=1 nohup python avg_model_lmc.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt mat52 none ../results --rank 4 > lmc_nohups/mat52_4.nohup &
OMP_NUM_THREADS=1 nohup python avg_model_lmc.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt mat52 none ../results --rank 5 > lmc_nohups/mat52_5.nohup &

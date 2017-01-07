CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=1 nohup python sk_ind.py ../../data/instances.txt ../../data/emotions.gold ../../embs/glove.6B.100d.txt ../results --gpu > nohups/sk.nohup &

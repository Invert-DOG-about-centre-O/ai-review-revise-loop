import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch, json
torch.set_num_threads(1)
import experiment as exp
r = exp.run_seed(500, epochs=14)
print(json.dumps(r['acc_seen']))
print(json.dumps([rec['sem_entropy'] for rec in r['records'][:5]]))

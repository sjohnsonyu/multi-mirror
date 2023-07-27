# 2x2, h=1
python3 double_oracle.py --n_perturb 0 --height 2 --width 2 --budget 2 --agent_train 50 --nature_train 50 --n_eval 20 --interval 2 --horizon 1 --max_epochs 2 --objective poaching --balance_attract 1 --write 0
python3 double_oracle.py --n_perturb 0 --height 2 --width 2 --budget 2 --agent_train 50 --nature_train 50 --n_eval 20 --interval 2 --horizon 1 --max_epochs 2 --objective logging --balance_attract 1 --write 0

# 2x2 Toy, h=1
python3 double_oracle.py --n_perturb 0 --height 2 --width 2 --budget 2 --agent_train 50 --nature_train 50 --n_eval 20 --interval 2 --horizon 1 --max_epochs 2 --objective poaching --balance_attract 1 --write 0 --toy 1
python3 double_oracle.py --n_perturb 0 --height 2 --width 2 --budget 2 --agent_train 50 --nature_train 50 --n_eval 20 --interval 2 --horizon 1 --max_epochs 2 --objective logging --balance_attract 1 --write 0 --toy 1

# 3x3 Toy, h=1
python3 double_oracle.py --n_perturb 0 --height 3 --width 3 --budget 5 --agent_train 50 --nature_train 50 --n_eval 20 --interval 2 --horizon 3 --max_epochs 2 --objective poaching --balance_attract 1 --write 0
python3 double_oracle.py --n_perturb 0 --height 3 --width 3 --budget 5 --agent_train 50 --nature_train 50 --n_eval 20 --interval 2 --horizon 3 --max_epochs 2 --objective logging --balance_attract 1 --write 0


# 5x5
python3 double_oracle.py --n_perturb 0 --height 5 --width 5 --budget 10 --agent_train 100 --nature_train 100 --n_eval 50 --interval 2 --horizon 3 --max_epochs 5 --objective logging --balance_attract 1 --write 0 
python3 double_oracle.py --n_perturb 0 --height 5 --width 5 --budget 10 --agent_train 100 --nature_train 100 --n_eval 50 --interval 2 --horizon 3 --max_epochs 5 --objective poaching --balance_attract 1 --write 0




# 10x10
python3 double_oracle.py --n_perturb 0 --height 10 --width 10 --budget 10 --agent_train 100 --nature_train 100 --n_eval 50 --interval 2 --horizon 3 --max_epochs 5 --objective poaching --balance_attract 1 --write 0
python3 double_oracle.py --n_perturb 0 --height 10 --width 10 --budget 10 --agent_train 100 --nature_train 100 --n_eval 50 --interval 2 --horizon 3 --max_epochs 5 --objective logging --balance_attract 1 --write 0
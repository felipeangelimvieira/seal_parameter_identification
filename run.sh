cd experiments
python3 prepare_dataset.py --data_path="amb_sin/*" --save_path=sin.csv --signal=sin
python3 prepare_dataset.py --data_path="amb_other/flat_*" --save_path=multisine.csv --signal=sweep
python3 prepare_dataset.py --data_path="amb_other/sweep*" --save_path=sweep.csv --signal=sweep

python3 models/run_model.py --model=linreg --data_path=experiments/sin.csv --save_path=experiments/results_linreg.csv --savgol
python3 models/run_model.py --model=gradient --data_path=experiments/sin.csv --save_path=experiments/results_gradient.csv --savgol
python3 models/run_model.py --model=eiv --data_path=experiments/sin.csv --save_path=experiments/results_eiv.csv 
python3 models/run_model.py --model=linreg_sweep --data_path=experiments/sweep.csv --save_path=experiments/results_linreg_sweep.csv  --savgol

python3 models/run_model.py --model=sin_projection_linreg --data_path=experiments/multisine.csv --save_path=experiments/results_linreg_multisine.csv --frequencies=[5,9,13,17,21,25,29,33,37,41,45,49,53,57,61,65,69]

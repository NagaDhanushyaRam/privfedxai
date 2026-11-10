@echo off
setlocal enabledelayedexpansion
rem DP sweep (D65): 3 repetitions x 3 epsilons, one run at a time. Log per run: results\fl\log_eps_E_r_R.txt
cd /d C:\Projects\privfedxai\fl_app
set PYTHONPATH=C:\Projects\privfedxai
for %%R in (1 2 3) do (
  for %%E in (10 5 1) do (
    echo !date! !time! start eps=%%E rep=%%R
    flwr run . --stream --federation-config="num-supernodes=5 client-resources-num-cpus=2" --run-config "dp-epsilon=%%E dp-rep=%%R" > ..\results\fl\log_eps%%E_r%%R.txt 2>&1
  )
)
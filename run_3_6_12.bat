call C:\Users\kuriy\Anaconda3\Scripts\\activate.bat
call conda activate M_2
REM python ef_run_3.py 2>&1 | Add-Content -Path "result_3/command.txt" -PassThru
mkdir "6_result"
python ef_run_6.py > "6_result/DO_COPY!!_command.txt"
REM 2>&1 | Add-Content -Path "6_result/DO_COPY!!_command.txt" -PassThru
mkdir "12_result"
python ef_run_12.py > "12_result/DO_COPY!!_command.txt"
REM 2>&1 | Add-Content -Path "6_result/DO_COPY!!_command.txt" -PassThru

pause
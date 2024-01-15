ps -ef | grep "PDCFRPlus/scripts/run.py" | grep -v grep | awk '{print $2}' | xargs kill

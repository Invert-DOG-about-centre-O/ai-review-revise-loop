#!/bin/bash
python experiment.py > blind_review_2_run_out.txt 2>&1
echo "EXIT:$?"
tail -20 blind_review_2_run_out.txt

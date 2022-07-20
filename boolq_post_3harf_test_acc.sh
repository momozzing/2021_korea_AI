#!/bin/bash

for epoch in $(seq 1 30);
do
    for step in $(seq 640 640 8420)
    do
        echo $var
        python boolq_submission_post_3harf.py --target_epochs=$epoch --target_step=$step
    done
done

# for var in {1..30}
# do
#     echo $var
#     python boolq_submission.py
# done



# python boolq_submission.py --target_epochs=2

# python boolq_submission.py --target_epochs=3 

# python boolq_submission.py --target_epochs=4 

# python boolq_submission.py --target_epochs=5 

# python boolq_submission.py --target_epochs=6 

# python boolq_submission.py --target_epochs=7 

# python boolq_submission.py --target_epochs=8

# python boolq_submission.py --target_epochs=9 

# python boolq_submission.py --target_epochs=10 

# python boolq_submission.py --target_epochs=11 

# python boolq_submission.py --target_epochs=12

# python boolq_submission.py --target_epochs=13 

# python boolq_submission.py --target_epochs=14 

# python boolq_submission.py --target_epochs=15

# python boolq_submission.py --target_epochs=16 

# python boolq_submission.py --target_epochs=16 

# python boolq_submission.py --target_epochs=17 

# python boolq_submission.py --target_epochs=18 

# python boolq_submission.py --target_epochs=19 

# python boolq_submission.py --target_epochs=20 

# python boolq_submission.py --target_epochs=21 

# python boolq_submission.py --target_epochs=22 

# python boolq_submission.py --target_epochs=23

# python boolq_submission.py --target_epochs=24 

# python boolq_submission.py --target_epochs=25 

# python boolq_submission.py --target_epochs=26 

# python boolq_submission.py --target_epochs=27 

# python boolq_submission.py --target_epochs=28 

# python boolq_submission.py --target_epochs=29 

# python WiC_submission.py --target_epochs=27 --random_seed=28
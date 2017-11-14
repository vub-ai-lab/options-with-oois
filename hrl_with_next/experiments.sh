#!/bin/bash
# Usage:
#   - set JOBS to the number of cores avaiable on the machine
#   - Replace "false" by "true" for all the experiments you want to launch
#   - Ensure that python3, gym, keras, theano, pyserial, python-opencv and zbar are installed
#   - run "./experiments.sh"
#   - Wait a bit. The results will be put in ???-results/variant/ directories
JOBS=4

# Modified DuplicatedInput
if false
then
    OMP_NUM_THREADS=1 parallel -j$JOBS --header : python3 options.py $2 \
        --name "100-0.001-{option}-{run}" \
        --env DuplicatedInputCond-v0 \
        --episodes 150000 \
        --avg 10 \
        --hidden 100 \
        --lr 0.001 \
        --option {option} \
        --random-nexts \
        ::: option 4 8 16 \
        ::: run 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20

    mkdir -p modifiedduplicatedinput-results/random/
    mv out* modifiedduplicatedinput-results/random/
fi

if false
then
    OMP_NUM_THREADS=1 parallel -j$JOBS --header : python3 options.py $2 \
        --name "100-0.001-{run}" \
        --env DuplicatedInputCond-v0 \
        --episodes 150000 \
        --avg 10 \
        --hidden 100 \
        --lr 0.001 \
        --option 2 \
        --subs "\"{-1: [20, 21], 0: range(20), 1: range(20)}\"" \
        --nexts "\"{0: [21], 1: [20, 21]}\"" \
        ::: run 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20

    mkdir -p modifiedduplicatedinput-results/next/
    mv out* modifiedduplicatedinput-results/next/
fi

if false
then
    OMP_NUM_THREADS=1 parallel -j$JOBS --header : python3 options.py $2 \
        --name "100-0.001-{run}" \
        --env DuplicatedInputCond-v0 \
        --episodes 150000 \
        --avg 10 \
        --hidden 100 \
        --lr 0.001 \
        --option 2 \
        --subs "\"{-1: [20, 21], 0: range(20), 1: range(20)}\"" \
        ::: run 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20

    mkdir -p modifiedduplicatedinput-results/nonext/
    mv out* modifiedduplicatedinput-results/nonext/
fi

if false
then
    OMP_NUM_THREADS=1 parallel -j$JOBS --header : python3 options.py $2 \
        --lstm \
        --name "100-0.001-{run}" \
        --env DuplicatedInputCond-v0 \
        --episodes 150000 \
        --avg 10 \
        --hidden 20 \
        --lr 0.001 \
        --option 2 \
        --subs "\"{-1: [20, 21], 0: range(20), 1: range(20)}\"" \
        ::: run 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20

    mkdir -p modifiedduplicatedinput-results/lstm_options/
    mv out* modifiedduplicatedinput-results/lstm_options/
fi

# Experiments on the Khepera
if false
then
    KERAS_BACKEND=theano python3 options.py $2 \
        --name khepera \
        --env Khepera-v0 \
        --sony \
        --episodes 100000 \
        --avg 10 \
        --hidden 100 \
        --lr 0.001 \
        --policy robot_policy.py
fi

# Simulated terminals
if false
then
    OMP_NUM_THREADS=1 parallel -j$JOBS --header : python3 options.py $2 \
        --name "100-0.001-{run}" \
        --env Terminals-v0 \
        --episodes 35000 \
        --avg 10 \
        --hidden 100 \
        --lr 0.001 \
        --option 12 \
        --subs "\"{-1: range(3, 15), 0: [0], 1: [0], 2: [0], 3: [0], 4: [1], 5: [2], 6: [1], 7: [2], 8: [1], 9: [2], 10: [1], 11: [2]}\"" \
        --nexts "\"{0: [7, 8], 1: [9, 10], 2: [11, 12], 3: [13, 14], 4: [3, 4], 5: [5, 6], 6: [3, 4], 7: [5, 6], 8: [3, 4], 9: [5, 6], 10: [3, 4], 11: [5, 6]}\"" \
        ::: run 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20

    mkdir -p terminals-results/next/
    mv out* terminals-results/next/
fi

if false
then
    OMP_NUM_THREADS=1 parallel -j$JOBS --header : python3 options.py $2 \
        --name "100-0.001-{run}" \
        --env Terminals-v0 \
        --episodes 35000 \
        --avg 10 \
        --hidden 100 \
        --lr 0.001 \
        --option 12 \
        --subs "\"{-1: range(3, 15), 0: [0], 1: [0], 2: [0], 3: [0], 4: [1], 5: [2], 6: [1], 7: [2], 8: [1], 9: [2], 10: [1], 11: [2]}\"" \
        ::: run 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20

    mkdir -p terminals-results/nonext/
    mv out* terminals-results/nonext/
fi

if false
then
    OMP_NUM_THREADS=1 parallel -j$JOBS --header : python3 options.py $2 \
        --lstm \
        --name "100-0.001-{run}" \
        --env Terminals-v0 \
        --episodes 35000 \
        --avg 10 \
        --hidden 20 \
        --lr 0.001 \
        --option 12 \
        --subs "\"{-1: range(3, 15), 0: [0], 1: [0], 2: [0], 3: [0], 4: [1], 5: [2], 6: [1], 7: [2], 8: [1], 9: [2], 10: [1], 11: [2]}\"" \
        ::: run 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20

    mkdir -p terminals-results/lstm_options/
    mv out* terminals-results/lstm_options/
fi

# TreeMaze
if false
then
    # Note for next4: there are 8 options defined here, but --subs only allows
    # allows the top-level policy to select options among a subset of 4.
    OMP_NUM_THREADS=1 parallel -j$JOBS --header : python3 options.py $2 \
        --name "100-0.001-{run}" \
        --env TreeMaze-v0 \
        --episodes 20000 \
        --avg 10 \
        --hidden 100 \
        --lr 0.001 \
        --option 8 \
        --subs "\"{-1: [3, 5, 7, 9], 0: range(3), 1: range(3), 2: range(3), 3: range(3), 4: range(3), 5: range(3), 6: range(3), 7: range(3)}\"" \
        --nexts "\"{0: [3, 4, 5, 7], 1: [4], 2: [5, 6], 3: [6], 4: [7, 8, 9], 5: [8], 6: [9, 10], 7: [10]}\"" \
        --policy treemaze_policy8.py \
        ::: run 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20

    mkdir -p treemaze-results/next4/
    mv out-* treemaze-results/next4/
fi

if false
then
    OMP_NUM_THREADS=1 parallel -j$JOBS --header : python3 options.py $2 \
        --name "100-0.001-{run}" \
        --env TreeMaze-v0 \
        --episodes 20000 \
        --avg 10 \
        --hidden 100 \
        --lr 0.001 \
        --option 8 \
        --subs "\"{-1: [3, 5, 7, 9], 0: range(3), 1: range(3), 2: range(3), 3: range(3), 4: range(3), 5: range(3), 6: range(3), 7: range(3)}\"" \
        --policy treemaze_policy8.py \
        ::: run 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20

    mkdir -p treemaze-results/nonext4/
    mv out-* treemaze-results/nonext4/
fi

if false
then
    OMP_NUM_THREADS=1 parallel -j$JOBS --header : python3 options.py $2 \
        --name "100-0.001-{run}" \
        --env TreeMaze-v0 \
        --episodes 20000 \
        --avg 10 \
        --hidden 100 \
        --lr 0.001 \
        --option 8 \
        --subs "\"{-1: range(3, 11), 0: range(3), 1: range(3), 2: range(3), 3: range(3), 4: range(3), 5: range(3), 6: range(3), 7: range(3)}\"" \
        --nexts "\"{0: [3, 4, 5, 7], 1: [4], 2: [5, 6], 3: [6], 4: [7, 8, 9], 5: [8], 6: [9, 10], 7: [10]}\"" \
        --policy treemaze_policy8.py \
        ::: run 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20

    mkdir -p treemaze-results/next8/
    mv out-* treemaze-results/next8/
fi

if false
then
    OMP_NUM_THREADS=1 parallel -j$JOBS --header : python3 options.py $2 \
        --name "100-0.001-{run}" \
        --env TreeMaze-v0 \
        --episodes 20000 \
        --avg 10 \
        --hidden 100 \
        --lr 0.001 \
        --option 8 \
        --subs "\"{-1: range(3, 11), 0: range(3), 1: range(3), 2: range(3), 3: range(3), 4: range(3), 5: range(3), 6: range(3), 7: range(3)}\"" \
        --policy treemaze_policy8.py \
        ::: run 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20

    mkdir -p treemaze-results/nonext8/
    mv out-* treemaze-results/nonext8/
fi

if false
then
    OMP_NUM_THREADS=1 parallel -j$JOBS --header : python3 options.py $2 \
        --name "100-0.001-{run}" \
        --env TreeMaze-v0 \
        --episodes 20000 \
        --avg 10 \
        --hidden 100 \
        --lr 0.001 \
        --option 14 \
        --subs "\"{-1: range(3, 17), 0: [2], 1: [2], 2: [2], 3: [2], 4: [2], 5: [2], 6: range(3), 7: range(3), 8: range(3), 9: range(3), 10: range(3), 11: range(3), 12: range(3), 13: range(3)}\"" \
        --nexts "\"{0: [5, 6], 1: [7, 8], 2: [9, 10], 3: [11, 12], 4: [13, 14], 5: [15, 16], 6: [9], 7: [10], 8: [11], 9: [12], 10: [13], 11: [14], 12: [15], 13: [16]}\"" \
        --policy treemaze_policy.py \
        ::: run 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20

    mkdir -p treemaze-results/next14/
    mv out-* treemaze-results/next14/
fi

if false
then
    OMP_NUM_THREADS=1 parallel -j$JOBS --header : python3 options.py $2 \
        --name "100-0.001-{run}" \
        --env TreeMaze-v0 \
        --episodes 20000 \
        --avg 10 \
        --hidden 100 \
        --lr 0.001 \
        --option 14 \
        --subs "\"{-1: range(3, 17), 0: [2], 1: [2], 2: [2], 3: [2], 4: [2], 5: [2], 6: range(3), 7: range(3), 8: range(3), 9: range(3), 10: range(3), 11: range(3), 12: range(3), 13: range(3)}\"" \
        --policy treemaze_policy.py \
        ::: run 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20

    mkdir -p treemaze-results/nonext14/
    mv out-* treemaze-results/nonext14/
fi

if false
then
    OMP_NUM_THREADS=1 parallel -j$JOBS --header : python3 options.py $2 \
        --lstm \
        --name "100-0.001-{run}" \
        --env TreeMaze-v0 \
        --episodes 100000 \
        --avg 10 \
        --hidden 100 \
        --lr 0.001 \
        --option 14 \
        --subs "\"{-1: range(3, 17), 0: [2], 1: [2], 2: [2], 3: [2], 4: [2], 5: [2], 6: range(3), 7: range(3), 8: range(3), 9: range(3), 10: range(3), 11: range(3), 12: range(3), 13: range(3)}\"" \
        --policy treemaze_policy.py \
        ::: run 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20

    mkdir -p treemaze-results/lstm14/
    mv out-* treemaze-results/lstm14/
fi

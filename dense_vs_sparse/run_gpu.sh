NUM_ROWS=2000
NUM_COLS=2000
NUM_TRIALS=15
FOLDER="rtx2060" # k20m, rtx3080

mkdir -p $FOLDER

# 2000 x 2000 dense with GPU
for run in $(seq 1 1 $NUM_TRIALS)
do
    python measure.py $NUM_ROWS $NUM_COLS dense cuda $FOLDER
done

# 2000 x 2000 csr with GPU
for run in $(seq 1 1 $NUM_TRIALS)
do
    python measure.py $NUM_ROWS $NUM_COLS csr cuda $FOLDER
done

# 2000 x 2000 csr with GPU
for run in $(seq 1 1 $NUM_TRIALS)
do
    python measure.py $NUM_ROWS $NUM_COLS auto cuda $FOLDER
done

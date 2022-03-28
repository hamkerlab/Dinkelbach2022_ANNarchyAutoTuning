NUM_ROWS=2000
NUM_COLS=2000
NUM_TRIALS=15
FOLDER="nvidia_RTX2060"

mkdir -p $FOLDER

# 2000 x 2000 dense with GPU
for run in $(seq 1 1 $NUM_TRIALS)
do
    python measure.py $NUM_ROWS $NUM_COLS dense cuda $TARGET
done

# 2000 x 2000 csr with GPU
for run in $(seq 1 1 $NUM_TRIALS)
do
    python measure.py $NUM_ROWS $NUM_COLS csr cuda $TARGET
done

# 2000 x 2000 csr with GPU
for run in $(seq 1 1 $NUM_TRIALS)
do
    python measure.py $NUM_ROWS $NUM_COLS auto cuda $TARGET
done

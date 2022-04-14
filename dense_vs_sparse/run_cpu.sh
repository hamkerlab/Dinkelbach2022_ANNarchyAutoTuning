NUM_ROWS=2000
NUM_COLS=2000
NUM_TRIALS=1
FOLDER="ryzen7"

mkdir -p $FOLDER

# 2000 x 2000 dense with SIMD
for run in $(seq 1 1 $NUM_TRIALS)
do
    python measure.py $NUM_ROWS $NUM_COLS dense openmp 0 $FOLDER
done

# 2000 x 2000 dense with no SIMD
for run in $(seq 1 1 $NUM_TRIALS)
do
    python measure.py $NUM_ROWS $NUM_COLS dense openmp 1 $FOLDER
done

# 2000 x 2000 csr with SIMD
for run in $(seq 1 1 $NUM_TRIALS)
do
    python measure.py $NUM_ROWS $NUM_COLS csr openmp 0 $FOLDER
done

# 2000 x 2000 csr with no SIMD
for run in $(seq 1 1 $NUM_TRIALS)
do
    python measure.py $NUM_ROWS $NUM_COLS csr openmp 1 $FOLDER
done

# 2000 x 2000 auto with SIMD
for run in $(seq 1 1 $NUM_TRIALS)
do
    python measure.py $NUM_ROWS $NUM_COLS auto openmp 0 $FOLDER
done

# 2000 x 2000 auto with no SIMD
for run in $(seq 1 1 $NUM_TRIALS)
do
    python measure.py $NUM_ROWS $NUM_COLS auto openmp 1 $FOLDER
done

#
#   Reads in a list of configurations stored in FNAME and starts
#   a measurement per data point
FNAME=configurations.csv
TARGET_FOLDER="../datasets/"
TARGET_NAME="nvidia_K20m.csv"

for i in $FNAME; do
    while read line; do
        python measure.py $line $TARGET_FOLDER $TARGET_NAME
    done < $FNAME;
done

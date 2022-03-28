#
#   Reads in a list of configurations stored in FNAME and starts
#   a measurement per data point
FNAME=article.csv
TARGET_FOLDER="../datasets/"
TARGET_NAME="nvidia_RTX2060.csv"

for i in $FNAME; do
    while read line; do
        echo $line
        python measure.py $line $TARGET_FOLDER $TARGET_NAME
    done < $FNAME;
done

#install nltk and scikit-learn

if test "$#" -ne 2; then
    echo "Illegal number of parameters"
else
    python mp2.py $1 $2
fi


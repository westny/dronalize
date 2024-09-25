python -m preprocessing.preprocess_urban --config 'rounD' --debug $1 --use-threads $2
python -m preprocessing.preprocess_urban --config 'inD' --debug $1 --use-threads $2
python -m preprocessing.preprocess_urban --config 'uniD' --debug $1 --use-threads $2
python -m preprocessing.preprocess_urban --config 'SIND' --debug $1 --use-threads $2
python -m preprocessing.preprocess_highway --config 'highD' --debug $1 --use-threads $2
python -m preprocessing.preprocess_highway --config 'exiD' --debug $1 --use-threads $2
python -m preprocessing.preprocess_highway --config 'A43' --debug $1 --use-threads $2
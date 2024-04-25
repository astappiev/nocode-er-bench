To test the **DeepMatcher** docker, use the following commands:

* eval "$(conda shell.bash hook)"
* conda activate deepmatcher

* git clone https://github.com/anhaidgroup/deepmatcher.git
* cd deepmatcher/examples/sample_data
* mv amz_goog_train.csv train.csv
* mv amz_goog_test.csv test.csv
* mv amz_goog_validation.csv val.csv

* git clone https://github.com/nishadi/deepmatcher-sample.git
* cd deepmatcher-sample
* python run.py /workspace/deepmatcher/examples/sample_data sample

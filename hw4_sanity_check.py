import numpy as np
import pickle
import csv
from imp import reload
from memm_tagger import viterbi_decode

label_vocab = dict()

# check viterbi with test inputs
def check_viterbi():

    with open("test.pickle","rb") as f:
        examples = pickle.load(f)
    # test_input = examples[0]["tensor"]

    try:
        # for example in examples:
        #     tensor, predictions = example["tensor"], example["predictions"]
        #     print(len(predictions))
        #     print(predictions)
        #     if viterbi_decode(example["tensor"]) != example["predictions"]:
        #         print("FAIL!")
        #         # import pdb; pdb.set_trace()
        #         print(example["tensor"].shape)
        #         print(len(example["predictions"]))
        #         import pdb; pdb.set_trace()
        #         viterbi_decode(example["tensor"])
        results = [viterbi_decode(example["tensor"]) == example["predictions"] for example in examples]
        # import pdb; pdb.set_trace()
        if len(set(results)) == 1 and results[0]:
            print("Check Successful")
        else:
            print("There are errors in your function")
    except Exception as e:
        print("Error!")
        print(e.args)

if __name__ == "__main__":
    check_viterbi()

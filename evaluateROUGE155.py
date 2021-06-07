import glob
import time
import os
import cvxpy
import numpy as np
from pyrouge import Rouge155
import random

random.seed(2019)
np.random.seed(2019)


def sentence_selector_ilp(lengths, weights, maxlen):
    selection = cvxpy.Variable(len(weights), boolean=True)
    length_constraint = lengths @ selection <= maxlen  # @ is for matrix multiplication
    if np.max(lengths) != 0:
        lenweight = np.multiply(np.divide(lengths, np.max(lengths)), weights)
    else:
        lenweight = weights
    total_utility = lenweight @ selection
    problem = cvxpy.Problem(cvxpy.Maximize(total_utility), [length_constraint])  # , selection>=0, selection<=1])
    problem.solve(solver=cvxpy.ECOS_BB)
    return selection.value


def calc_rouge(candidates, references, temp_dir, method):
    assert len(candidates) == len(references)

    cnt = len(candidates)
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    if not os.path.isdir(temp_dir):
        os.mkdir(temp_dir)
    tmp_dir = os.path.join(temp_dir, method)
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
    tmp_dir = os.path.join(temp_dir, "rouge-tmp-{}".format(current_time))
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    try:

        for i in range(cnt):
            if len(references[i]) < 1:
                continue
            with open(tmp_dir + "/candidate/cand.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(candidates[i])
            with open(tmp_dir + "/reference/ref.{}.txt".format(i), "w",
                      encoding="utf-8") as f:
                f.write(references[i])
        r = Rouge155()  # temp_dir=temp_dir)
        r.model_dir = tmp_dir + "/reference/"
        r.system_dir = tmp_dir + "/candidate/"
        r.model_filename_pattern = 'ref.#ID#.txt'
        r.system_filename_pattern = r'cand.(\d+).txt'
        rouge_results = r.convert_and_evaluate()

        print(method, rouge_results)
        results_dict = r.output_to_dict(rouge_results)
    finally:
        pass
        # if os.path.isdir(tmp_dir):
        #     shutil.rmtree(tmp_dir)
    return results_dict


def test_rouge(input, mode='test', tune_param=False):
    method, temp_dir, use_ilp = input[0], input[1], input[2]
    print(method, str(os.getpid()))
    candidates = []
    references = []
    if tune_param:
        range_low = 0
        range_high = 4000
    else:
        range_low = 4250
        range_high = 4500
    for i in range(range_low, range_high):
        print('data/' + mode + '/' + str(i) + '.sents.txt')
        story_file = glob.glob('data/' + mode + '/' + str(i) + '.sents.txt')[0]
        sentences = [line.strip() for line in open(story_file, 'r').readlines()]
        # read slides
        pptfile = 'slide_generator_data/data/' + str(i) + '/clean_tika.txt'
        slides = [line.strip() for line in open(pptfile, 'r').readlines()]
        reference = ' '.join(slides)

        limit = int(0.2 * len(' '.join(sentences)))

        label_file = glob.glob('data/' + mode + '/' + str(i) + method)[0]
        scores = [float(label.strip()) for i, label in enumerate(open(label_file, 'r').readlines()[:len(sentences)])]
        lengths = [len(sent) for sent in sentences]
        if tune_param:
            # this part is to evaluate the oracle summaries built by different window sizes
            candidate = ' '.join([sentences[i] for i, score in enumerate(scores) if score == 1])[:limit]
        else:
            if use_ilp:
                try:
                    selections = sentence_selector_ilp(np.array(lengths), np.array(scores), limit)
                    selections = np.rint(selections)
                    candidate = ' '.join([sentences[i] for i, indicator in enumerate(selections) if indicator == 1])
                except:
                    lines = [(label, i) for i, label in enumerate(scores)]
                    lines.sort(key=lambda x: (-x[0], x[1]))
                    candidate = ' '.join([sentences[line[1]] for line in lines])[:limit]
            else:
                lines = [(label, i) for i, label in enumerate(scores)]
                lines.sort(key=lambda x: (-x[0], x[1]))
                candidate = ' '.join([sentences[line[1]] for line in lines])[:limit]

        references.append(reference)
        candidates.append(candidate)
    print('*********** candidate len and reference len:::', len(candidates), len(references))
    return calc_rouge(candidates, references, temp_dir, method)


methods = [("_predicted_SummaRunnerScores.txt", "temp", True),
           ("_windowed_predicted_SummaRunnerScores.txt", "temp", True),
           ]

from multiprocess import Pool

pool = Pool(10)
results = pool.map(test_rouge, methods)
for result, method in zip(results, methods):
    print('---------', method[0], method[2], ':', result)
pool.close()
pool.join()

import os
import subprocess
import json
from wordrep.utils import BaseRepDetector


def eval_score(directory='./corpus/', ref_path='../corpus/rep_test.fr'):
    c = "sacrebleu"  # Replace with your desired Bash command
    rep_detector = BaseRepDetector('fr')
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            # Process the file
            if filename.endswith('fr'):
                path = os.path.join(directory,filename)
                command = f'{c} {path} -i {ref_path}'
                # Execute the command
                process = subprocess.Popen(
                    command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # Wait for the command to finish and capture the output
                output, _ = process.communicate()
                # Decode the output
                output = output.decode('utf-8')
                o = json.loads(output)
                reps = rep_detector.detect_corpus(path, vis=False)
                print(filename, f"bleu={o['score']}, rep={len(reps)}")

if __name__ == '__main__':
    eval_score()
import re

def contains_only_punctuation(text):
    pattern = r'^[^\w\s]*$'
    return re.match(pattern, text) is not None

# Test cases
text1 = "!@#$%^&*()"
text2 = "Hello, world!"
text3 = "こんにちは"
text4 = "Élève"

print(contains_only_punctuation(text1))  # True
print(contains_only_punctuation(text2))  # False
print(contains_only_punctuation(text3))  # False
print(contains_only_punctuation(text4))  # False

import subprocess
from wordrep.utils import BaseRepDetector
import json
filename = '../corpus/rep_test.en.out.avg'
command = f"sacrebleu {filename} -i ../corpus/rep_test.fr"  # Replace with your desired Bash command
# Execute the command
process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
# Wait for the command to finish and capture the output
output, error = process.communicate()
# Decode the output
output = output.decode('utf-8')
o = json.loads(output)
rep_detector = BaseRepDetector('fr')
reps = rep_detector.detect_corpus(filename, vis=False)
print(filename, f"bleu={o['score']}, rep={len(reps)}")
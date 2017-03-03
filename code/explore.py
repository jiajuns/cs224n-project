'''
Find out the max length of context and question, so that we know what is the max length to pad the sentence
'''
from __future__ import print_function
import matplotlib.pyplot as plt

context_max_len = 0
context_len = []
with open('data/squad/train.ids.context', 'r') as file:
    for line in file:
        context_max_len = max(context_max_len, len(line))
        context_len.append(len(line))
with open('data/squad/val.ids.context', 'r') as file:
    for line in file:
        context_max_len = max(context_max_len, len(line))
        context_len.append(len(line))
print("Max Sentence Length for context is {}".format(str(context_max_len)))

question_max_len = 0
question_len = []
with open('data/squad/train.ids.question', 'r') as file:
    for line in file:
        question_max_len = max(question_max_len, len(line))
        question_len.append(len(line))
with open('data/squad/val.ids.question', 'r') as file:
    for line in file:
        question_max_len = max(question_max_len, len(line))
        question_len.append(len(line))
print("Max Sentence Length for cuestion is {}".format(str(question_max_len)))


num_bins = 50

# the histogram of the data
fig, ax = plt.subplots()
n, bins, patches = ax.hist(context_len, num_bins)
plt.axvline(context_max_len, color='k', linestyle='solid')
ax.set_xlabel('Context Length')
ax.set_ylabel('Count')
ax.set_title('Histogram of Context Length')
plt.savefig('plots/context_hist.png')
# the histogram of the data
fig, ax = plt.subplots()
n, bins, patches = ax.hist(question_len, num_bins)
plt.axvline(question_max_len, color='k', linestyle='solid')
ax.set_xlabel('question Length')
ax.set_ylabel('Count')
ax.set_title('Histogram of Question Length')
plt.savefig('plots/question_hist.png')


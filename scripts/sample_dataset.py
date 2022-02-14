from tqdm import trange
import random
import sys

if len(sys.argv) != 8:
    print('USAGE: {} <source_file> <source_file_number_of_lines> <output_train_file> <validation_size> <output_validation_file> <test_size> <output_test_file>'.format(sys.argv[0]))
    print('e.g.: {} source.jsonl 90000000 train.jsonl 10000 valid.jsonl 10000 test.jsonl'.format(sys.argv[0]))
    sys.exit(1)

source_file = sys.argv[1]
source_size = int(sys.argv[2])
train_file = sys.argv[3]
valid_size = int(sys.argv[4])
valid_file = sys.argv[5]
test_size = int(sys.argv[6])
test_file = sys.argv[7]

fd = open(source_file, 'r')
train_fd = open(train_file, 'w')
test_fd = open(test_file, 'w')
valid_fd = open(valid_file, 'w')

total = source_size

sample = random.sample(list(range(0, total)), valid_size + test_size)

valid_idx = sample[:valid_size]
test_idx = sample[valid_size:]

valid_idx.sort()
test_idx.sort()

assert len(valid_idx) == valid_size
assert len(test_idx) == test_size

valid_iter = iter(valid_idx)
test_iter = iter(test_idx)

valid_i = next(valid_iter)
test_i = next(test_iter)

for x in trange(0, total):
    line = fd.readline()
    if not line:
        print('Reached end of file')
        break
    if x == valid_i:
        valid_fd.write(line)
        try:
            valid_i = next(valid_iter)
        except StopIteration:
            pass
    elif x == test_i:
        test_fd.write(line)
        try:
            test_i = next(test_iter)
        except StopIteration:
            pass
    else:
        train_fd.write(line)

fd.close()
valid_fd.close()
train_fd.close()
test_fd.close()

assert valid_i == valid_idx[-1] and test_i == test_idx[-1], 'Error the size of the validation/test file is not as requested.'

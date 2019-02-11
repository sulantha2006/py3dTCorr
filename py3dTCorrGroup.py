#!/data/data05/sulantha/PythonProjects/py3dTCorr/.env/bin/python
import nibabel
from nilearn.masking import apply_mask
import numpy
from multiprocessing import Pool
import h5py
from scipy.stats import pearsonr

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--datafile', required=True, help='File list CSV. No header, No other columns. ')
parser.add_argument('--mask', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--nthreads', required=False, default=24, type=int)
parser.add_argument('--block_size', required=False, default=100, type=int)
args = vars(parser.parse_args())

data_file = args['datafile']
mask_file = args['mask']
out_ = args['out']

arrays = [nibabel.load(i).get_fdata() for i in open(data_file, 'r').readlines()]
data_ = numpy.stack(arrays, axis=0)

masked_data = apply_mask(data_, nibabel.load(mask_file))
masked_data = masked_data.astype(numpy.float32)

## This was a sanity check - test with small data set
#masked_data = numpy.random.randint(100, size=(53, 237))

nthreads = args['nthreads']
pool_count = nthreads
b_size = args['block_size']

def get_corr(tups):
    return [pearsonr(tup[0], tup[1])[0] for tup in tups]


triu_indices = numpy.triu_indices(masked_data.shape[1], 1, masked_data.shape[1])


def createGen(batch_count, parallelsize, block_size=1):
    start = batch_count*parallelsize*block_size
    end = (batch_count+1)*parallelsize*block_size
    if end > triu_indices[0].shape[0]:
        end = triu_indices[0].shape[0]
    for i in range(start, end, block_size):
        yield [(masked_data[:,triu_indices[0][i+k]], masked_data[:,triu_indices[1][i+k]]) for k in range(block_size) if i+k < end]


pool = Pool(processes=nthreads)

total_count = int(masked_data.shape[1]*(masked_data.shape[1]-1)/2)
batch_count = int(numpy.ceil(total_count/pool_count/b_size))

result = numpy.zeros(total_count, dtype=numpy.float32)

print('Starting analysis...', flush=True)
result_index = 0
for i in range(batch_count):
    if i % 100 == 0:
        print('Batch {0}/{1}: {2:.2f}%'.format(i+1, batch_count, (i+1)*100/batch_count), flush=True)
    generator = createGen(i, pool_count, b_size)
    corr_struct_ = pool.map(get_corr, generator)
    corr_struct_ = [val for sublist in corr_struct_ for val in sublist]
    result[result_index:result_index+len(corr_struct_)] = corr_struct_
    result_index += len(corr_struct_)
print('Finished analysis.', flush=True)

## This was a sanity check
#tcorr = numpy.corrcoef(masked_data, rowvar=False)
#tcorr = tcorr[triu_indices]

#print(numpy.allclose(tcorr, result))
### Sanity finished.

print('Corr array dimensions: {0}'.format(result.shape), flush=True)
print('Writing result ...', flush=True)
h5f = h5py.File(out_, 'w')
h5f.create_dataset('corr', data=result)
h5f.close()
print('Complete. ', flush=True)

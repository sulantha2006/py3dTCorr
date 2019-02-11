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
parser.add_argument('--labels', required=True)
parser.add_argument('--out', required=True)
parser.add_argument('--nthreads', required=False, default=24, type=int)
parser.add_argument('--block_size', required=False, default=100, type=int)
args = vars(parser.parse_args())

data_file = args['datafile']
mask_file = args['mask']
label_file = args['labels']
out_ = args['out']

mask_file_data = nibabel.load(mask_file)
reshape_size = mask_file_data.get_fdata().shape

fixed_mask_mat = numpy.round(mask_file_data.get_fdata(), decimals=0)
mask_file_data = nibabel.Nifti1Image(fixed_mask_mat, mask_file_data.affine, mask_file_data.header)

arrays = [nibabel.load(i).get_fdata() for i in open(data_file, 'r').readlines()]
data_ = numpy.stack(arrays, axis=0)

masked_data = apply_mask(data_, mask_file_data)
masked_data = masked_data.astype(numpy.float32)

label_file_data = apply_mask([nibabel.load(label_file)], mask_file_data)
label_file_data = label_file_data.astype(numpy.int_)
label_file_data_unique = numpy.unique(label_file_data)
label_file_data_unique = numpy.delete(label_file_data_unique, numpy.where(label_file_data_unique==0))

print('Found {0} unique values in label file - {1}. '.format(label_file_data_unique.shape[0], list(label_file_data_unique)), flush=True)

def get_label_values(_data, _label_data, _labels):
    out = numpy.zeros((_data.shape[0], _labels.shape[0]))
    for idx in range(_labels.shape[0]):
        out[:, idx] = numpy.squeeze(numpy.mean(_data[:, numpy.where((_labels[idx] - 0.25 < _label_data[0]) & (_label_data[0] < _labels[idx] + 0.25))], axis=-1, keepdims=True))
    return out

## This was a sanity check - test with small data set
#masked_data = numpy.random.randint(100, size=(53, 237))

nthreads = args['nthreads']
pool_count = nthreads
b_size = args['block_size']

def get_corr(tups):
    return [pearsonr(tup[0], tup[1])[0] for tup in tups]


#_indices = numpy.triu_indices(masked_data.shape[1], 1, masked_data.shape[1])
_indices = numpy.arange(masked_data.shape[1])

label_values = get_label_values(masked_data, label_file_data, label_file_data_unique)

def createGen(batch_count, parallelsize, label_idx, block_size=1):
    start = batch_count*parallelsize*block_size
    end = (batch_count+1)*parallelsize*block_size
    if end > _indices.shape[0]:
        end = _indices.shape[0]
    for i in range(start, end, block_size):
        yield [(label_values[:, label_idx], masked_data[:, _indices[i + k]]) for k in range(block_size) if i + k < end]


pool = Pool(processes=nthreads)

total_count = int(masked_data.shape[1])
batch_count = int(numpy.ceil(total_count/pool_count/b_size))

result = numpy.zeros((label_values.shape[1], total_count), dtype=numpy.float32)

print('Starting analysis...', flush=True)

for lab_idx in range(label_values.shape[1]):
    print('Label - {0}'.format(lab_idx+1))
    result_index = 0
    for i in range(batch_count):
        if i % 100 == 0:
            print('Batch {0}/{1}: {2:.2f}%'.format(i+1, batch_count, (i+1)*100/batch_count), flush=True)
        generator = createGen(i, pool_count, lab_idx, b_size)
        corr_struct_ = pool.map(get_corr, generator)
        corr_struct_ = [val for sublist in corr_struct_ for val in sublist]
        result[lab_idx, result_index:result_index+len(corr_struct_)] = corr_struct_
        result_index += len(corr_struct_)
print('Finished analysis.', flush=True)

## This was a sanity check
#tcorr = numpy.corrcoef(masked_data, rowvar=False)
#tcorr = tcorr[triu_indices]

#print(numpy.allclose(tcorr, result))
### Sanity finished.

print('Corr array dimensions: {0}'.format(result.shape), flush=True)
print('Writing result to h5...', flush=True)
h5f = h5py.File(out_+'.h5', 'w')
h5f.create_dataset('corr', data=result)
h5f.close()
print('Writing result to images...', flush=True)
for idx in range(result.shape[0]):
    dat = result[idx, :]
    new_array = numpy.zeros(reshape_size, dtype=numpy.float32)
    new_array[fixed_mask_mat==1] = dat
    vn = nibabel.Nifti1Image(new_array, mask_file_data.affine, mask_file_data.header)
    vn.to_filename(out_ + '_' + str(idx) + '.nii')

print('Complete. ', flush=True)

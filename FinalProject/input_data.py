import glob
import numpy as np
from tqdm import tqdm
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes

TRAIN_PATH = 'dataset/train'

def extractLabels(path):
    number = 0
    array = np.zeros((4))
    number = int(path[14:15])
    array[number] = 1
    return array
def extractImages(f):
    return map(lambda x: x/255, np.loadtxt(f))
def getFiles(path):
    files = glob.glob('{}/**/*.txt*'.format(path))
    images = []
    labels = []
    for f in tqdm(files):
        try:
            image = extractImages(f)
            label = extractLabels(f)
            labels.append(label)
            images.append(image)
            # image = np.array(midi_manipulation.midiToNoteStateMatrix(f))
            # if np.array(song).shape[0] > 50:
                # songs.append(song)
        except Exception as e:
            raise e           
    return [np.array(images).reshape(24, 307200), np.array(labels).reshape(24, 4)]



def _read32(bytestream):
  dt = np.dtype(np.uint32).newbyteorder('>')
  return np.frombuffer(bytestream.read(4), dtype=dt)[0]


# def extract_images(f):
#       """Extract the images into a 4D uint8 np array [index, y, x, depth].

#   Args:
#     f: A file object that can be passed into a gzip reader.

#   Returns:
  
#     data: A 4D uint8 np array [index, y, x, depth].

#   Raises:
#     ValueError: If the bytestream does not start with 2051.

#   """
#   print('Extracting', f.name)
#   with gzip.GzipFile(fileobj=f) as bytestream:
#     magic = _read32(bytestream)
#     if magic != 2051:
#       raise ValueError('Invalid magic number %d in MNIST image file: %s' %
#                        (magic, f.name))
#     num_images = _read32(bytestream)
#     rows = _read32(bytestream)
#     cols = _read32(bytestream)
#     buf = bytestream.read(rows * cols * num_images)
#     data = np.frombuffer(buf, dtype=np.uint8)
#     data = data.reshape(num_images, rows, cols, 1)
#     return data





# def extract_labels(f, one_hot=False, num_classes=10):
#   """Extract the labels into a 1D uint8 numpy array [index].

#   Args:
#     f: A file object that can be passed into a gzip reader.
#     one_hot: Does one hot encoding for the result.
#     num_classes: Number of classes for the one hot encoding.

#   Returns:
#     labels: a 1D uint8 numpy array.

#   Raises:
#     ValueError: If the bystream doesn't start with 2049.
#   """
#   print('Extracting', f.name)
#   with gzip.GzipFile(fileobj=f) as bytestream:
#     magic = _read32(bytestream)
#     if magic != 2049:
#       raise ValueError('Invalid magic number %d in MNIST label file: %s' %
#                        (magic, f.name))
#     num_items = _read32(bytestream)
#     buf = bytestream.read(num_items)
#     labels = np.frombuffer(buf, dtype=np.uint8)
#     if one_hot:
#       return dense_to_one_hot(labels, num_classes)
#     return labels


class DataSet(object):

  def __init__(self,
               images,
               labels,
               fake_data=False,
               one_hot=False,
               dtype=dtypes.float32,
               reshape=True):
    """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.
    """
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
    self._num_examples = images.shape[0]
  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=0):
  train_images, train_labels = getFiles(train_dir)
  if not 0 <= validation_size <= len(train_images):
    raise ValueError(
        'Validation size should be between 0 and {}. Received: {}.'
        .format(len(train_images), validation_size))
#   validation_images = train_images[:validation_size]
#   validation_labels = train_labels[:validation_size]
  train_images = train_images[validation_size:]
  train_labels = train_labels[validation_size:]
  train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
#   validation = DataSet(validation_images,
#                        validation_labels,
#                        dtype=dtype,
#                        reshape=reshape)
#   test = DataSet(test_images, test_labels, dtype=dtype, reshape=reshape)

  # return base.Datasets(train=train)
  return train
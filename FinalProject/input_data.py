import glob
import numpy as np

def extractLabels(path):
    return path[14:15]

def getFiles(path):
    files = glob.glob('{}/**/*.txt*'.format(path))
    images = []
    for f in files:
        try:
            content = map(lambda x: x/255, np.loadtxt(f))
            image = [content, extractLabels(f)]
            images.append(image)
            print(len(content))
            # image = np.array(midi_manipulation.midiToNoteStateMatrix(f))
            # if np.array(song).shape[0] > 50:
                # songs.append(song)
        except Exception as e:
            raise e           
    return images

getFiles('dataset/train')
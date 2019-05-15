from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as pl
from skimage import external
import numpy as np
import warnings
import skimage
from pathlib import Path
import pandas as pd
import numpy as np
from skimage import img_as_ubyte

from imagestitcher import stitcher
warnings.filterwarnings('ignore')


def makePathRecords(path):
    tileRecords = []
    for f in path.iterdir():
        if '.tif' in f.name:
            names = ['fullpath', 'name', 'index', 'desc']
            desc, strIndex, source = f.name.split('_')
            tileRecords.append(dict(zip(names, [str(f), f.name, int(strIndex), desc])))
    rasterData = pd.DataFrame.from_dict(tileRecords).sort_values('index').reset_index(drop = True)
    return rasterData


def stitchOrgRaster(rasterDataDF, stitchParams,ncols):
    nrows = len(rasterDataDF) // ncols
    splitArr = np.array_split(rasterDataDF.iloc[0:ncols*nrows].fullpath.tolist(), nrows)

    rows = []
    for index in tqdm(range(nrows), desc = 'Stitching Rows'):
        chunk = splitArr[index]
        fr = stitcher.FlatRaster(chunk, stitchParams)
        fr.fetchImages()
        if index%2==0:
            fr.params.acquiOri = (0, 1)
        else:
            fr.params.acquiOri = (1, 0)
        rows.append(fr.cutStitchRect(stitchParams.imsize))

    arrangedTiles = np.concatenate(np.array(rows), axis = 0) #Stitch rows
    return arrangedTiles


def getFactors(rasterTileDF, start = 20):
    numfiles = len(rasterTileDF.index.unique())
    factors = []
    for n in range(20, numfiles):
        m = numfiles/n
        if m == float(int(m)):
            factors.append(tuple(sorted([n, int(numfiles/n)])))
    return set(factors)
    
def saveRaster(imgArr, outPath, compressionLevel = 9):
    if compressionLevel:
        external.tifffile.imsave(str(outPath), imgArr, compress = compressionLevel)
    else:
        external.tifffile.imsave(str(outPath), imgArr, compress = None)
        
def readImage(path):
    return external.tifffile.imread(str(path))

def resizeImage(imgarr, scaleFactor, numPartitions = 50):
    nrows = numPartitions # Split the image into chunks to make the resizing operations easier
    splitImg = np.array_split(imgarr, nrows)

    rows = []
    for index in tqdm(range(nrows), desc = 'Resizing Partitions'):
        toResize = splitImg[index]
        smallImg = skimage.transform.resize(toResize, (toResize.shape[0]//scaleFactor, toResize.shape[1]//scaleFactor))
        rows.append(smallImg)

    return img_as_ubyte(np.concatenate(np.array(rows), axis = 0)) #Stitch rows


def sliceImage(wellSlices, exportPath, scaledImage, fullImage, scaleFactor, desc):

    maxwidth = scaledImage.shape[1]
    maxheight = scaledImage.shape[0]
    sf = scaleFactor #scale factor
    wellslices = wellSlices
    well_indices = range(len(wellslices))
    makeSlice = lambda rangeVals, sf: slice(*[d*sf for d in rangeVals])

    indices_slices = dict(zip(well_indices, [(makeSlice(hs, sf), makeSlice(vs, sf)) for hs, vs in wellslices]))
    common_name = exportPath

    slices = [{'well': i, 'hstart': s[0].start, 'hstop': s[0].stop, 'vstart': s[1].start, 'vstop': s[1].stop} for i, s in indices_slices.items()]
    with open(common_name.joinpath(Path('imageSlices.csv')), 'a+') as record:
        record.write('#numwells\t{}\n#description\t{}\n#scaleFactor\t{}\n'.format(str(len(wellslices)), str(desc), str(scaleFactor)))
        pd.DataFrame.from_dict(slices).to_csv(record)

    for index, slices in tqdm(indices_slices.items(), desc = 'Exporting Well Image'):
        savePath = common_name.joinpath(Path('well_{}.tif'.format(str(index))))
        saveRaster(fullImage[slices[0], slices[1]], str(savePath))
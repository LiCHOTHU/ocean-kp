import json 
import glob
import os
import numpy as np

class loader(object):
    """docstring for loader"""
    def __init__(self, type_, binSize, endTimestamp):
        self.type_ = type_
        self.endTimestamp = endTimestamp
        self.binSize = binSize

    def smooth_reward_curve(self, x, y):
        halfwidth = min(31, int(np.ceil(len(x)/30))) # Halfwidth of our smoothing convolution
        k = halfwidth
        xsmoo = x[k:-k]
        ysmoo = np.convolve(y, np.ones(2*k+1), mode='valid') / np.convolve(np.ones_like(y), np.ones(2*k+1), mode='valid')
        downsample = max(int(np.floor(len(xsmoo)/1e3)),1)
        return xsmoo[::downsample], ysmoo[::downsample]

    def fix_point(self, x, y):
        np.insert(x, 0, 0)
        np.insert(y, 0, 0)

        fx, fy, index = [], [], 0
        ninterval = int(max(x) / self.binSize + 1)
        for i in range(ninterval):
            tmpx = self.binSize * i
            while index + 1 < len(x) and tmpx > x[index + 1]:
                index += 1

            if (index + 1) < len(x):
                alpha = (y[index+1] - y[index]) / (x[index+1] - x[index])
                tmpy = y[index] + alpha * (tmpx - x[index])
                fx.append(tmpx)
                fy.append(tmpy)

        return np.array(fx), np.array(fy)

    def load_one_dir(self, indir, batchsize=1):
        data = []
        infiles = glob.glob(os.path.join(indir, '*monitor.json'))

        for inf in infiles:
            with open(inf, 'r') as f:
                t_start = float(json.loads(f.next())['t_start'])
                for line in f:
                    tmp = json.loads(line)
                    data.append([float(tmp['t']) + t_start, int(tmp['l']), float(tmp['r'])])

        # Sort by timestamp
        data = np.array(sorted(data, key=lambda d_entry:d_entry[0]))
        endIndex = np.where(np.cumsum(data[:,1]) < self.endTimestamp)[0][-1]
        timesteps = np.cumsum(data[:endIndex,1])
        y = data[:endIndex, -1]

        if self.type_ == 'timesteps':
            x = timesteps / 4
        elif self.type_ == 'time':
            x = data[:endIndex, 0] - data[0, 0]
        elif self.type_ == 'eposide':
            x = np.array(range(len(y)))
        elif self.type_ == 'updates':
            x = timesteps / batchsize

        x, y = self.smooth_reward_curve(x, y)
        x, y = self.fix_point(x, y)

        if self.type_ == 'time':
            x = x.astype(float) / 3600
        return [x, y]

    def load(self, indir):
        dirs = glob.glob(os.path.join(indir, '*/'))
        result = []

        for i in range(len(dirs)):
            tmpx, tmpy = [], []
            label = dirs[i].strip('/').split('/')[-1]
            indirs = glob.glob(os.path.join(dirs[i], '*/'))

            batchsize = -1
            if self.type_ == 'updates' and '_' in label:
                batchsize = int(label.split('_')[-1])
            if self.type_ == 'eposide':
                batchsize = 640

            for indir in indirs:
                if batchsize > 0:
                    tx, ty = self.load_one_dir(indir, batchsize)
                else:
                    tx, ty = self.load_one_dir(indir)

                tmpx.append(tx)
                tmpy.append(ty)

            if len(tmpx) > 1:
                length = max([len(t) for t in tmpx])
                longest = None
                for j in range(len(tmpx)):
                    if len(tmpx[j]) == length:
                        longest = tmpx[j]

                # For line with less data point, the last value will be repeated and appended
                # Until it get the same size with longest one
                for j in range(len(tmpx)):
                    if len(tmpx[j]) < length:
                        repeaty = np.ones(length - len(tmpx[j])) * tmpy[j][-1]
                        addindex = len(tmpx[j]) - length
                        addx = longest[addindex:]
                        tmpy[j] = np.append(tmpy[j], repeaty)
                        tmpx[j] = np.append(tmpx[j], addx)

                x = np.mean(np.array(tmpx), axis=0)
                y_mean = np.mean(np.array(tmpy), axis=0)
                y_std = np.std(np.array(tmpy), axis=0)
            else:
                x = np.array(tmpx).reshape(-1)
                y_mean = np.array(tmpy).reshape(-1)
                y_std = np.zeros(len(y_mean))

            result.append([label, x, y_mean, y_std])
        return result

    
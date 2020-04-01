import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.getcwd(), os.pardir))
from plot import colorPanel, loader, stick
from plot.cd import cd
import glob
import json
import csv
import itertools
#from plotting import cd
#from plotting import baseline_logger as bl

matplotlib.rcParams.update({'font.size': 16})

def smooth_reward_curve(x, y):
    # print (len(x))
    halfwidth = min(151, int(np.ceil(len(x)/15))) # Halfwidth of our smoothing convolution
    k = halfwidth
    xsmoo = x[k:-k]
    ysmoo = np.convolve(y, np.ones(2*k+1), mode='valid') / np.convolve(np.ones_like(y), np.ones(2*k+1), mode='valid')
    downsample = max(int(np.floor(len(xsmoo)/1e3)),1)
    return xsmoo[::downsample], ysmoo[::downsample]

def process_json(path, sstr):
    with open(path, 'r') as f:
        r = {'l': [], 'r': []}
        ccnt = 0
        kk = 0
        ff = 1
        fs = csv.reader(f)
        for line in fs:
            if ff:
                ff = 0
                idx = line.index('AverageReturn_all_%s_tasks'%sstr)
                continue
            kk += 1
            if line[idx] == 'nan':
                break
            r['l'].append(kk)
            r['r'].append(float(line[idx]))
    for k in ['l', 'r']:
        r[k] = np.array(r[k])
    # print (len(r['l']))
    if len(r['l']) <= 50:
        return None
    return r


def average_dict(res, keys, er, rr):
    """
    :param res: the dict
    :param keys: keys
    :return: a dict with (key, (mean, std)) of results.
    """
    ret = dict()
    for k in keys:
        vs = [r[k] for r in res]
        l = min(v.shape[0] for v in vs)
        vs = [v[np.newaxis, :l] for v in vs]
        vs = np.concatenate(vs, axis=0)
        # if k == 'r':
        #     vs = (vs-rr)/(er-rr)
        m, s = np.median(vs, axis=0), np.std(vs, axis=0, dtype=np.float64)
        ret[k] = [m, s]
    return ret

def get_data(prefix, criterion, er, rr, sstr):

    with cd(prefix):
        paths = glob.glob(criterion)
        d = dict()
        if len(paths) == 0:
            return None
        assert len(paths) == 1
        print ("find path:", paths)
        for path in paths:
            files = glob.glob(path + '/seed-*/*/progress.csv')
            # files = glob.glob(path + '/*/progress.csv')
            res = []
            if len(files) == 0:
                return None
            for f in files:
                root_f = f[:-12]
                if len(glob.glob(os.path.join(root_f, "online-train-epoch799.pkl"))) == 0:
                    continue
                r = process_json(f, sstr)
                if r == None:
                    continue
                res.append(r)
            print (len(res))
            if True:
                res = average_dict(res, ['l', 'r'], er=er, rr=rr)
            d[path] = res
        # print ("len is %d"%(len(paths)))
        if len(paths) == 1:
            for k, v in d.items():
                return v
        else:
            return d


def plotArg(game, ax, maxy, sstr):
    # env_id, traj_need = game.split('-')
    # traj_need = int(traj_need)
    colors = colorPanel.colorPanel(1).getColors()
    # print (len(colors)) len is 10
    dims = ['0', '4'] # ['0', '4']
    ncats = ['0', '2', '5'] # ['5', '10']
    cdims = ['2', '4', '5'] #, '2', '4', '5']
    ddims = ['0', '4'] # ['4', '0']
    ndirs = ['2', '5', '0'] # ['5', '10']
    lams = ['50.0']
    tems = ['0.33'] #, '0.67']
    anns = ['False']
    units = ['False']
    dcs = ['0.0'] #, '5.0'] # ['5.0'] # ['0.0', '5.0']
    ccs = ['0.0'] #, '5.0']
    dics = ['0.0'] #, '5.0']
    als = ['None', '0.7']
    cs = ['logitnormal', 'None']
    eas = ['None'] #, '0.1']
    varss = ['None', '3.0', '5.0', '7.0', '9.0']
    cnt = 0

    expert_reward = -6.22
    random_reward = -13.46

    # criterion = 'dim-%s-ncat-%s-cdim-%s-ddim-%s-ndir-%s-lam-%s-tem-%s-ann-%s-unit-%s-dc-%s-cc-%s-dic-%s-a-%s-c-%s-ea-%s-var-%s/'\
    #                 %(dim, ncat, cdim, ddim, ndir, lam, tem, ann, unit, dc, cc, dic, al, c, ea, var)
    num = 0
    for dim, ncat, cdim, ddim, ndir, lam, tem, ann, unit, dc, cc, dic, al, c, ea, var in \
            itertools.product(dims, ncats, cdims, ddims, ndirs, lams, tems, anns, units, dcs, ccs, dics, als, cs, eas, varss):
        # criterion = 'dim-%s-cat-%s-cdim-%s-lam-%s-tem-%s-ann-%s-unit-%s-dc-%s-cc-%s/'\
        #                 %(dim, ncat, cdim, lam, tem, ann, unit, dc, cc)
        criterion = 'dim-%s-ncat-%s-cdim-%s-ddim-%s-ndir-%s-lam-%s-tem-%s-ann-%s-unit-%s-dc-%s-cc-%s-dic-%s-a-%s-c-%s-ea-%s-var-%s/'\
                        %(dim, ncat, cdim, ddim, ndir, lam, tem, ann, unit, dc, cc, dic, al, c, ea, var)
        # criterion = 'dim-%s-ncat-%s-cdim-%s-ddim-%s-ndir-%s-lam-%s-tem-%s-ann-%s-unit-%s-dc-%s-cc-%s-dic-%s-a-%s-c-%s-var-%s/'\
        #                 %(dim, ncat, cdim, ddim, ndir, lam, tem, ann, unit, dc, cc, dic, al, c, var)
        # print (criterion)
        datas = get_data(prefix='../dir-new/%s/'%game, \
                     criterion=criterion, er = expert_reward, rr = random_reward, sstr=sstr)
        # datas = get_data(prefix='/test/oyster/dir-new/%s/'%game, \
        #              criterion=criterion, er = expert_reward, rr = random_reward)

        label = 'dim-%s-ncat-%s-cdim-%s-ddim-%s-ndir-%s-var-%s'%(dim, ncat, cdim, ddim, ndir, var)
        # label = criterion
        # continue

        if datas == None:
            continue

        # print (label)
        # continue

        num += 1

        x = datas['l']
        y = datas['r']

        x = x[0]
        y_mean = y[0]
        y_std = y[1]

        xx = x
        x, y_mean = smooth_reward_curve(xx, y_mean)
        x, y_std = smooth_reward_curve(xx, y_std)

        cnt += 1
        color = colors[cnt%10]

        y_upper = y_mean + 0.05*y_std
        y_lower = y_mean - 0.05*y_std
        ax.fill_between(
            x, list(y_lower), list(y_upper), interpolate=True, facecolor=color, linewidth=0.0, alpha=0.3
        )
        if cnt >= 10:
            line = ax.plot(x, list(y_mean), label=label, linestyle='--', color=color, rasterized=True)
        else:
            line = ax.plot(x, list(y_mean), label=label, color=color, rasterized=True)
        # break
    print ('find %s runs'%num)
    # ax.set_xticks([])
    # ax.set_yticks([])
    # game = 'Cooperative Communication'
    # stick.cutsomStick(game, 'timesteps', ax)
    # if maxy != None:
    #     ax.set_ylim(0, maxy)
    # if args == 'Reacher':
    #     ax.set_ylim(-10, 0)
    # ax.set_xticks([750])
    # ax.set_yticklabels(['1'])
    # ax.set_yticks([ax.get_yticks()[-1]])
    # print (ax.get_yticks()[-1])
    # print (plt.xticks(), plt.yticks()[-1])


if __name__ == '__main__':
    # fig = plt.figure()
    fig = plt.figure(figsize=(8,8))

    task_name = [
                    # 'reach-v1', 'push-v1', 'pick-place-v1', 'door-open-v1', 'drawer-close-v1', \
                    # 'button-press-topdown-v1', 'peg-insert-side-v1', 'window-open-v1', 'sweep-v1', 'basketball-v1', \
                    # 'drawer-open-v1', 'door-close-v1', 'shelf-place-v1', 'sweep-into-v1', 'lever-pull-v1'
                    'meta10'
                ]
    task_id = 0
    sixAtariGames = [
        # 'sparse-dirichlet-point-robot',
        # 'sparse-direction-point-robot',
        # 'categorical-point-robot',
        # 'categorical-40',
        # 'meta/pick-place-v1'
        # 'sparse-dirichlet-robot-ori',
        # 'sparse-dirichlet-robot-sac',
        'meta/'+task_name[task_id]
    ]
    sstr = 'train'
    idxs = range(len(task_name))
    # idxs = [i for i in range(len(task_name)) if i % 2 == 0]
    # idxs = [2]
    sixAtariGames = ['meta/'+task_name[i] for i in idxs]
    maxy = None
    # title = ['original', 'updated sac']
    # title = ['updated sac']
    title = [task_name[i] for i in idxs]

    # ax = fig.add_subplot(111)
    columns = 1
    for i, args in enumerate(sixAtariGames):
        print(args)
        ax = fig.add_subplot(len(sixAtariGames) / columns + 1, columns, i + 1)
        ax.ticklabel_format(axis='y', style='sci')
        # plt.tick_params(
        #     axis='both',          # changes apply to the x-axis
        #     which='both',      # both major and minor ticks are affected
        #     bottom=False,      # ticks along the bottom edge are off
        #     top=False,         # ticks along the top edge are off
        #     labelbottom=False, # labels along the bottom edge are off
        # ) 
        # ax = fig.add_subplot(1, 1, 1)
        plotArg(args, ax, maxy, sstr)
        ax.set_xlabel('Timesteps', fontsize=13)
        ax.set_ylabel('Rewards', fontsize=13)
        plt.title(title[i]+'-%s'%sstr, fontsize=15)

    # plt.legend(loc=4)
    if len(idxs) > 1:
        lgd = plt.legend(loc='lower center', bbox_to_anchor=(-2.4, -1), ncol=2, fontsize=12)
    else:
        lgd = plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1, fontsize=10)
    fig.tight_layout()
    print ('save pdf')
    if len(idxs) > 1:
        fig.savefig('figs/test-update.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
    else:
        fig.savefig('figs/test-update-%s-%s.pdf'%(task_name[idxs[0]], sstr), bbox_extra_artists=(lgd,), bbox_inches='tight')
    # fig.savefig('test.pdf', dpi=1000, bbox_extra_artists=(lgd,), bbox_inches='tight')
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
import pickle
#from plotting import cd
#from plotting import baseline_logger as bl

matplotlib.rcParams.update({'font.size': 16})

def smooth_reward_curve(x, y):
    print (len(x))
    halfwidth = min(151, int(np.ceil(len(x)/15))) # Halfwidth of our smoothing convolution
    k = halfwidth
    xsmoo = x[k:-k]
    ysmoo = np.convolve(y, np.ones(2*k+1), mode='valid') / np.convolve(np.ones_like(y), np.ones(2*k+1), mode='valid')
    downsample = max(int(np.floor(len(xsmoo)/1e3)),1)
    return xsmoo[::downsample], ysmoo[::downsample]

def process_json(path, sstr, maxlen=-1):
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
            if line[idx] == 'nan' or kk==maxlen:
                break
            r['l'].append(kk)
            r['r'].append(float(line[idx]))
    for k in ['l', 'r']:
        r[k] = np.array(r[k])
    # print (len(r['l']))
    # if len(r['l']) <= 50:
    #     return None
    return r

def process_data(path, sstr, maxlen=-1):
    all_files = os.listdir(path)
    all_files = [i for i in all_files if 'data-epoch' in i]
    all_files = sorted(all_files, key=lambda x: int(x[10:-4]))
    kk = 0
    r = {'l': [], 'online-train-succ': [], 'online-test-succ': [], 'posterior-train-succ': []}
    for i in all_files:
        with open(os.path.join(path, i), 'rb') as f:
            cur_d = pickle.load(f)
        for j in cur_d:
            if j in r:
                r[j].append(np.array(cur_d[j]).flatten())
        r['l'].append(kk)
        kk += 1
        if kk == maxlen:
            break
    for k in r:
        r[k] = np.array(r[k])
        # print (k, r[k].shape)
    # if len(r['l']) <= 50:
    #     return None
    return r

def average_high_dict(res):
    ret = dict()
    keys = res[0].keys()
    for k in keys:
        vs = [r[k] for r in res]
        l = min(v.shape[0] for v in vs)
        vs = [v[np.newaxis, :l] for v in vs]
        vs = np.concatenate(vs, axis=0)
        # if k == 'r':
        #     vs = (vs-rr)/(er-rr)
        m, s = np.median(vs, axis=0), np.std(vs, axis=0, dtype=np.float64)
        if k != 'l':
            # m = np.reshape(m, [m.shape[0], -1, 20])
            # s = np.reshape(s, [s.shape[0], -1, 20])
            m = np.mean(m, axis=-1)
            s = np.mean(s, axis=-1)
        ret[k] = [m, s]
    return ret


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

#dim-0-ncat-0-cdim-0-ddim-5-ndir-3-lam-50.0-tem-0.33-ann-False-unit-False-dc-0.0-cc-0.0-dic-0.0-a-0.7-c-logitnormal-ea-None-var-5.0-vrnn-4.0.0.0.0-rnn-vrnn-vc-None-va-None-vvar-None
def get_data(prefix, criterion, er, rr, sstr, maxlen=-1, gather=False):

    with cd(prefix):
        paths = glob.glob(criterion)
        d, new_d = dict(), dict()
        if len(paths) == 0:
            return None
        assert len(paths) == 1
        print ("find path:", paths)
        for path in paths:
            files = glob.glob(path + '/seed-*/*/progress.csv')
            # files = glob.glob(path + '/*/progress.csv')
            res, new_res = [], []
            if len(files) == 0:
                return None
            for f in files:
                root_f = f[:-12]
                # print (sorted(os.listdir(root_f)))
                if len(glob.glob(os.path.join(root_f, "data-epoch10.pkl"))) == 0:
                    continue
                r = process_json(f, sstr, maxlen=maxlen)
                if gather:
                    new_r = process_data(root_f, sstr, maxlen=maxlen)
                if r == None:
                    continue
                res.append(r)
                if gather:
                    new_res.append(new_r)
            print (len(res))
            if True:
                res = average_dict(res, ['l', 'r'], er=er, rr=rr)
                if gather:
                    new_res = average_high_dict(new_res)
            d[path] = res
            if gather:
                new_d[path] = new_res
            else:
                new_d[path] = None
        # print ("len is %d"%(len(paths)))
        if len(paths) == 1:
            return [d[paths[0]], new_d[paths[0]]]
            # for k, v in d.items():
                # return v
        else:
            assert False
            return d


def plotArg(game, ax, maxy, sstr):
    # env_id, traj_need = game.split('-')
    # traj_need = int(traj_need)
    colors = colorPanel.colorPanel(1).getColors()
    # print (len(colors)) len is 10
    dims = ['0'] #, '4'] # ['0', '4']
    ncats = ['0'] #, '2', '5'] # ['5', '10']
    cdims = ['0', '1', '4'] #, '2', '4', '5']
    ddims = ['0', '3'] # ['4', '0']
    ndirs = ['2', '0'] # ['5', '10']
    lams = ['10.0']
    tems = ['0.33']
    anns = ['False']
    units = ['False']
    dcs = ['0.0'] #, '5.0'] # ['5.0'] # ['0.0', '5.0']
    ccs = ['0.0'] #, '5.0']
    dics = ['0.0'] #, '5.0']
    als = ['None', '0.7']
    cs = ['logitnormal', 'None']
    eas = ['None'] #, '0.1']
    varss = ['None', '5.0']

    vrnns = ['0.0.0.2.3', '4.0.0.0.0', '0.2.3.0.0', '0.0.0.0.0']
    # vrnns = ['0.2.4.0.0', '0.0.0.2.4', '0.0.0.0.0']
    rnns = ['rnn', 'None'] # ['vrnn', 'rnn', 'None']
    vcs = ['dirichlet', 'None']
    vas = ['None', '0.8']
    vvars = ['None']
    cnt = 0
    maxlen = -1
    gather = False

    expert_reward = -6.22
    random_reward = -13.46

    # criterion = 'dim-%s-ncat-%s-cdim-%s-ddim-%s-ndir-%s-lam-%s-tem-%s-ann-%s-unit-%s-dc-%s-cc-%s-dic-%s-a-%s-c-%s-ea-%s-var-%s/'\
    #                 %(dim, ncat, cdim, ddim, ndir, lam, tem, ann, unit, dc, cc, dic, al, c, ea, var)
    num = 0
    all_data = {}
    for dim, ncat, cdim, ddim, ndir, lam, tem, ann, unit, dc, cc, dic, al, c, ea, var, vrnn, rnn, vc, va, vvar in \
            itertools.product(dims, ncats, cdims, ddims, ndirs, lams, tems, anns, units, dcs, ccs, dics, als, cs, eas, varss, vrnns, rnns, vcs, vas, vvars):
        # criterion = 'dim-%s-cat-%s-cdim-%s-lam-%s-tem-%s-ann-%s-unit-%s-dc-%s-cc-%s/'\
        #                 %(dim, ncat, cdim, lam, tem, ann, unit, dc, cc)
        # criterion = 'dim-%s-ncat-%s-cdim-%s-ddim-%s-ndir-%s-lam-%s-tem-%s-ann-%s-unit-%s-dc-%s-cc-%s-dic-%s-a-%s-c-%s-ea-%s-var-%s-vrnn-%s-rnn-%s-vc-%s-va-%s-vvar-%s/'\
        #                 %(dim, ncat, cdim, ddim, ndir, lam, tem, ann, unit, dc, cc, dic, al, c, ea, var, vrnn, rnn, vc, va, vvar)
        criterion = 'dim-%s-ncat-%s-cdim-%s-ddim-%s-ndir-%s-lam-%s-tem-%s-ann-%s-unit-%s-dc-%s-cc-%s-dic-%s-a-%s-c-%s-var-%s-vrnn-%s-rnn-%s-vc-%s-va-%s-vvar-%s/'\
                        %(dim, ncat, cdim, ddim, ndir, lam, tem, ann, unit, dc, cc, dic, al, c, var, vrnn, rnn, vc, va, vvar)
        # print (criterion)
        datas = get_data(prefix='../rnn-new/%s/'%game, \
                     criterion=criterion, er = expert_reward, rr = random_reward, sstr=sstr, maxlen=maxlen, gather=gather)
        # datas = get_data(prefix='/test/oyster/dir-new/%s/'%game, \
        #              criterion=criterion, er = expert_reward, rr = random_reward)

        label = 'dim-%s-ncat-%s-cdim-%s-ddim-%s-ndir-%s-var-%s-vrnn-%s-%s-%s'%(dim, ncat, cdim, ddim, ndir, var, vrnn, rnn, vc)
        # label = criterion
        # continue

        if datas == None:
            continue
        # if label in ['dim-0-ncat-0-cdim-1-ddim-0-ndir-0-var-None-vrnn-0.2.4.0.0-vrnn-dirichlet', \
        #              'dim-0-ncat-0-cdim-1-ddim-0-ndir-0-var-None-vrnn-0.0.0.2.4-vrnn-dirichlet', \
        #              'dim-0-ncat-0-cdim-1-ddim-0-ndir-0-var-None-vrnn-0.0.0.2.4-vrnn-logitnormal']:
        #     continue
        all_data[label] = datas
        if gather:
            datas = datas[1]
        else:
            datas = datas[0]
        num += 1
        # print (label)
        # continue

        x = datas['l']
        if gather:
            y = datas['posterior-train-succ']
        else:
            y = datas['r']

        x = x[0]
        y_mean = y[0]
        y_std = y[1]

        xx = x
        # print (len(xx), len(y_mean))
        if True:
            x, y_mean = smooth_reward_curve(xx, y_mean)
            x, y_std = smooth_reward_curve(xx, y_std)
        else:
            y_std = 0
        # print (len(xx), len(y_mean))
        # exit(-1)

        cnt += 1
        color = colors[cnt%10]

        y_upper = y_mean + 0.2*y_std
        y_lower = y_mean - 0.2*y_std
        ax.fill_between(
            x, list(y_lower), list(y_upper), interpolate=True, facecolor=color, linewidth=0.0, alpha=0.3
        )
        if cnt >= 10:
            line = ax.plot(x, list(y_mean), label=label, linestyle='--', color=color, rasterized=True)
        else:
            line = ax.plot(x, list(y_mean), label=label, color=color, rasterized=True)
        # break
    print ('find %s runs'%num)
    ax.set_xticks([])
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
    fig = plt.figure(figsize=(14,8))

    task_name = [
                    'reach-v1', 'push-v1', 'pick-place-v1', 'door-open-v1', 'drawer-close-v1', \
                    'button-press-topdown-v1', 'peg-insert-side-v1', 'window-open-v1', 'sweep-v1', 'basketball-v1', \
                    'drawer-open-v1', 'door-close-v1', 'shelf-place-v1', 'sweep-into-v1', 'lever-pull-v1', \
                    'meta10', 'cheetah-multi-vel', 'cheetah-multi-gra'
                ]
    task_id = -1
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
    # idxs = range(len(task_name))
    # idxs = [i for i in range(len(task_name)) if i % 2 == 0]
    idxs = [-1]
    sixAtariGames = [task_name[i] for i in idxs]
    # sixAtariGames = ['meta/'+task_name[i] for i in idxs]
    # sixAtariGames = ['sparse-dirichlet-point-robot']
    # sixAtariGames = ['meta/']

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
        # ax.set_ylabel('Rewards', fontsize=13)
        plt.title(title[i]+'-%s'%sstr, fontsize=15)

    # plt.legend(loc=4)
    if len(idxs) > 1:
        lgd = plt.legend(loc='lower center', bbox_to_anchor=(-2.4, -1), ncol=2, fontsize=12)
    else:
        lgd = plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.7), ncol=2, fontsize=12)
        # pass
        # lgd = plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=1, fontsize=10)
    fig.tight_layout()
    print ('save pdf')
    if len(idxs) > 1:
        fig.savefig('figs/rnn-update-update-ml1-%s.pdf'%sstr, bbox_extra_artists=(lgd,), bbox_inches='tight')
    else:
        # fig.savefig('figs/rnn-update-%s.pdf'%task_name[idxs[0]])
        fig.savefig('figs/rnn-update-update-%s.pdf'%task_name[idxs[0]], bbox_extra_artists=(lgd,), bbox_inches='tight')

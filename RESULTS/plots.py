import numpy as np
import matplotlib.pyplot as plt


T = 49
formats = ['.-b', '*-r', '+-k', 'v-g', 's-c' 'd-y']


def plot_reliability(leadtimes, thresholds, observations, samples, labels):

    bins = np.arange(0.0, 1.1, 0.1)
    probs = np.arange(0.05, 1.0, 0.1)

    for leadt in leadtimes:
        for thresh in thresholds:
            print('Leadt:', leadt, ', Thresh:', thresh)
            hi = (observations[..., leadt] >= thresh).numpy()

            prob = []
            dig = []
            rel = []
            hist = []

            for sample in samples:
                temp_prob = (sample[..., leadt] >= thresh).mean(0)
                temp_dig  = probs[np.digitize(temp_prob,bins[1:], right=True)]
                temp_rel  = np.array([hi[temp_dig == prob].mean() for prob in probs])    
                temp_hist = np.histogram(temp_prob, bins)[0] / len(observations)
                temp_hist = np.insert(temp_hist, [0,10], 0.0)

                prob.append(temp_prob)
                dig.append(temp_dig)
                rel.append(temp_rel)
                hist.append(temp_hist)

            fig, axs = plt.subplots(2, 1, figsize=(5,8), gridspec_kw={'height_ratios': [3, 1]})

            axs[0].plot([0.0, 1.0], [0.0,1.0], 'k--', lw=1)
            for i in range(len(rel)):
                axs[0].plot(probs, rel[i], 'o-', label=labels[i])
            axs[0].set_xlim(0.0,1.0)
            axs[0].set_ylim(0.0,1.0)
            axs[0].set_aspect(1)
            axs[0].legend()
            axs[0].set_xlabel('Prob. forecast')
            axs[0].set_ylabel('Obs. freq.')
            axs[0].set_title(f'Lead time {leadt}h \n{thresh} m/s threshold')

            for i in range(len(hist)):
                axs[1].hist(prob[i], bins, density=True, histtype='step', log=True)
            axs[1].set_xlabel('Prob. forecast')
            axs[1].set_ylabel('Log density')
            #axs[1].set_yticks([])                      # uncomment this if ticks on y-axis should not be shown
            #axs[1].minorticks_off()                    # uncomment this if ticks on y-axis should not be shown


            plt.show()




def get_climatology(thresholds, leadtimes, obs):

    climatology = np.zeros((len(leadtimes), len(thresholds)))

    for k, ldt in enumerate(leadtimes):
        for t, thresh in enumerate(thresholds):
            climatology[k, t] = np.where(obs[:,ldt] >= thresh, 1, 0).mean()

    return climatology


def plot_bss(leadtimes, climatology, observations, samples, labels):

    thresh = np.arange(1.0, 21.0, 1.0)

    #climatology = get_climatology(thresh, leadtimes, observations)

    for leadt in leadtimes:
        print('Leadt:', leadt)
        hi = (observations[..., None, leadt] >= thresh).numpy()

        clim_bs = climatology[leadt, :len(thresh)]
        clim_bs = (clim_bs - hi)**2
        clim_bs = clim_bs.mean(0)

        bs = []

        for sample in samples:
            temp_bs = (sample[..., None, leadt] >= thresh).mean(0)
            temp_bs = (temp_bs - hi)**2
            temp_bs =  temp_bs.mean(0)

            bs.append(temp_bs)

        plt.hlines(0.0,1,20, lw=1, ls='dashed', color='k', label='Climat.')
        for i in range(len(bs)):
            plt.plot(thresh, 1 - bs[i]/clim_bs, label=labels[i])
        plt.xlim([1,20])
        plt.legend()
        plt.ylabel('BSS')
        plt.xlabel('Windspeed threshold (m/s)')
        plt.title(f'Lead time {leadt}h')
        ax = plt.gca()
        ax.figure.set_size_inches(5, 3)
        plt.show()




def plot_hist(leadtime, samples, bins, title=''):
    plt.figure()
    plt.hist(samples, bins=bins, color='skyblue', edgecolor='black', density=True)

    plt.xlabel('Values')
    plt.ylabel('Frequency')
    if title == '':
        plt.title('Leadtime ' + str(leadtime))
    else:
        plt.title(title)

    plt.show()


def plot_fitting(timesteps, log_step, values, labels, title, formats=formats, ylabel='Score'):
    plt.figure()
    for i in range(len(values)):
        plt.plot(np.arange(0, timesteps, log_step), values[i], formats[i], label=labels[i])
    plt.title(title)
    plt.xlabel('Minimization step')
    plt.xticks(np.arange(0, timesteps, 500))
    plt.xlim([0, timesteps])
    plt.ylabel(ylabel)
    plt.legend()


def plot_leadtimes(values, labels, title, formats=formats):
    plt.figure()
    for i in range(len(values)):
        plt.plot(np.arange(0,T), values[i], formats[i], label=labels[i])
    plt.title(title)
    plt.xlabel('Lead time')
    plt.xticks([0, 6, 12, 18, 24, 30, 36, 42, 48])
    plt.xlim([0, 48])
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_samples(samples, obs, fcst,  title='Samples'):
    plt.figure()
    for i in range(len(samples)):
        plt.plot(np.arange(0,T),samples[i])
    plt.plot(np.arange(0,T), fcst, '--k', linewidth=3, label='NWP Forecast')
    plt.plot(np.arange(0,T), obs, '-k', linewidth=3, label='Observation')
    plt.title(title)
    plt.xlabel('Lead time')
    plt.xticks([0, 6, 12, 18, 24, 30, 36, 42, 48])
    plt.xlim([0, 48])
    plt.ylabel('Wind Speed')
    plt.legend()
    plt.show()









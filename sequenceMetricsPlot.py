#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 15:07:36 2025

@author: loon
"""

# %%Visualization
def sns_pairplot(df, columns = None):
    import seaborn as sns
    import matplotlib as plt
    #sns.set(style="ticks", color_codes=True)
    #columns = ['nFrames', 'active_ratio','displacement', 'distance','pix_growth', 'pca_meanLL', 'pca_varex', 'tortuousity','angle_norm']
    if columns is None:
        g = sns.pairplot(df, plot_kws=dict(s=5, alpha=0.5))
    else:
        g = sns.pairplot(df,vars=columns, plot_kws=dict(s=5, alpha=0.5))


def act_overImagingSession(df, fxxx, minutes_list = None, ax=None):
    '''
    # TODO
        make this standalone and not rely on minutes list

    this plots the number of events per minutes over an imaging session
    minutes_list is especially necessary if imaging sessions have different times
        (can be computed from pull_event_sequences)
    but minutes_list will also provide a way to compare event rates over animals easier

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    minutes_list : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    '''

    x = df.tser.value_counts().sort_index().index
    y = np.array(df.tser.value_counts().sort_index())

    print(x)
    print(y)

    y_mins = []
    if minutes_list is not None:
        for y_, x_ in zip(y,x):
            y_mins.append(y_/((minutes_list[1,minutes_list[0,:]==x_])[0]))
    print(y_mins)

    if ax==None:
        fig, ax = plt.subplots(1,1)
        ax.plot(x,y_mins, label=f'F{fxxx.ferret}')
        ax.set_title(f'F{fxxx.ferret} Activity')
        ax.set_xlabel('T series')
        ax.set_ylabel('Sequences/minute')
    else:
        ax.plot(x,y_mins, label=f'F{fxxx.ferret}')

    return ax


def rose_plot(data, ferret=None, ax=None, histtype='bar', metric_classifiers=None, metric_names=[], colors=None):
    '''
    data is generally trajectory angles
    '''

    if ax is None:
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], polar=True)
        ax_flag = True
    else:
        ax_flag = False

    #ax.set_theta_zero_location('W')
    ax.set_theta_direction(-1)

    if metric_classifiers is None:
        h = np.histogram(data, bins=64, density=True)
        theta = np.deg2rad(h[1][1:])
        radii = h[0]

        if histtype=='bar':
            bars = ax.bar(theta, radii, width=2*np.pi/32, bottom=0.0)
        elif histtype=='step':
            steps = ax.step(theta, radii)

    else:
        for i, (value, name) in enumerate(zip(np.unique(metric_classifiers), metric_names)):
            h = np.histogram(data[metric_classifiers==value], bins=32)
            theta = np.deg2rad(h[1][1:])
            radii = h[0]
            if colors is not None:
                steps = ax.step(theta, radii, label=name, color=colors[i])
            else:
                steps = ax.step(theta, radii, label=name)

        plt.legend(loc='upper right')






    if ferret is not None:
        ax.set_title(f'F{ferret} Trajectories')
    else:
        ax.set_title('Trajectories')
    # for r,bar in zip(radii, bars):
    #     bar.set_facecolor( cm.magma(r/10.))
    #     bar.set_alpha(0.5)

    if ax_flag:
        return fig, ax
    else:
        return ax


def plot_log_regression(df, metric):
    from scipy.stats import linregress
    import matplotlib.pyplot as plt
    plt.figure()
    counts, bins, bars = plt.hist(df[metric], bins = 60)
    Regression=linregress(np.log(counts+1), np.log(bins[1:]))
    slope=Regression.slope
    plt.title(f"Slope of log-log hist: {slope}")


def plot_stationary_bars(df, ferret):
    total_counts = np.ones(df.nFrames.nunique())
    bars_stationary = df.groupby('nFrames').stationary_bool.mean()
    total_counts -= bars_stationary
    bars_indeterminate = np.zeros(df.nFrames.nunique())
    bars_indeterminate[df.nFrames.unique()<5] = total_counts[df.nFrames.unique()<5]
    total_counts -= bars_indeterminate
    bars_linear = df.groupby('nFrames').sig.mean()
    bars_linear[df.nFrames.unique()<5] = 0
    bars_nonlinear = total_counts - bars_linear

    fig, ax = plt.subplots(2,1, sharex=True)
    ax[0].bar(df.nFrames.unique(), bars_stationary, label='stationary')
    ax[0].bar(df.nFrames.unique(), np.zeros(df.nFrames.nunique()), label='--propagation--', facecolor='w', alpha=0)
    ax[0].bar(df.nFrames.unique(), bars_linear, label='linear', bottom=bars_stationary)
    ax[0].bar(df.nFrames.unique(), bars_nonlinear, label='nonlinear', bottom=(bars_stationary+bars_linear))
    ax[0].bar(df.nFrames.unique(), bars_indeterminate, label='indeterminate', bottom=(bars_stationary+bars_linear+bars_nonlinear))

    ax[0].legend()
    ax[0].set_ylabel('Proportion')

    total_counts = df.groupby('nFrames').count().tser
    bars_stationary = df.groupby('nFrames').stationary_bool.sum()
    total_counts -= bars_stationary
    bars_indeterminate = np.zeros(df.nFrames.nunique())
    bars_indeterminate[df.nFrames.unique()<5] = total_counts[df.nFrames.unique()<5]
    total_counts -= bars_indeterminate
    bars_linear = df.groupby('nFrames').sig.sum()
    bars_linear[df.nFrames.unique()<5] = 0
    bars_nonlinear = total_counts - bars_linear

    ax[1].bar(df.nFrames.unique(), bars_stationary, label='stationary')
    ax[1].bar(df.nFrames.unique(), np.zeros(df.nFrames.nunique()), label='--propagation--', facecolor='w', alpha=0)
    ax[1].bar(df.nFrames.unique(), bars_linear, label='linear', bottom=bars_stationary)
    ax[1].bar(df.nFrames.unique(), bars_nonlinear, label='nonlinear', bottom=(bars_stationary+bars_linear))
    ax[1].bar(df.nFrames.unique(), bars_indeterminate, label='indeterminate', bottom=(bars_stationary+bars_linear+bars_nonlinear))

    ax[1].legend()
    ax[1].set_ylabel('Count')
    ax[1].set_xlabel('nFrames')

    ax[0].set_title(f'F{ferret} Sequence Distribution by Frame')


# %% a very specific function for running significance



def quick_r2_plot():
    ferrets = [261,317,335,336,337,339]
    fig, ax = plt.subplots(3,2)
    ax = ax.flatten()

    for i in range(6):
        df_unsig = dataframes[i][dataframes[i]['sig']==False]
        ax[i].hist(dataframes_sig[i]['r2'], bins=32, histtype='step', density=True)
        ax[i].hist(df_unsig['r2'], bins=32, histtype='step', density=True)
        ax[i].set_title(f'F{ferrets[i]}')
        ax[i].set_xlabel('r2')
        if i==5:
            plt.legend(['Sig', 'Not Sig'])

    plt.tight_layout()


def quick_r2_nframe_plot():
    ferrets = [261,317,335,336,337,339]
    fig, ax = plt.subplots(3,2)
    ax = ax.flatten()

    for i in range(6):
        df = dataframes[i][dataframes[i]['sig']==True]
        ax[i].hist2d(df['r2'], df['nFrames'], density=True)
        ax[i].set_title(f'F{ferrets[i]}')
        ax[i].set_xlabel('r2')
        ax[i].set_ylabel('nFrames')

    plt.tight_layout()


def quick_roseplot_angleshuffle_compare():
    ferrets = [261,317,335,336,337,339]
    fig, ax = plt.subplots(3,2, subplot_kw={'projection': 'polar'})
    ax = ax.flatten()

    for i, ft in enumerate(ferrets):
        shuffled_angles = load_dataframe(ft)['shuffle_angles']
        shuffled_angles = [s for s in shuffled_angles if s is not None]
        shuffled_angles = np.concatenate(shuffled_angles)

        df_ = load_dataframe(ft)['df']

        ax[i] = rose_plot(df_[df_.sig==True].angle, ferret=ft, ax=ax[i], histtype='step')
        ax[i] = rose_plot(shuffled_angles, ferret=ft, ax=ax[i], histtype='step')

    ax[-1].legend(['Sig', 'Shuffles'])
    plt.tight_layout()


def bunch_of_roses():
    ferrets = [261,317,335,336,337,339]

    for ft in ferrets:
        df_ = load_dataframe(ft)['df']

        fig, ax = plt.subplots(1,4, subplot_kw={'projection': 'polar'}, figsize=(13,4))
        rose_plot(df_.angle, ferret=ft, ax=ax[0])
        rose_plot(df_[df_.sig==True].angle, ferret=ft, ax=ax[1])
        rose_plot(df_[df_.sig!=True].angle, ferret=ft, ax=ax[2])

        ax[3] = rose_plot(df_[df_.sig==True].angle, ferret=ft, ax=ax[3], histtype='step')
        ax[3] = rose_plot(df_[df_.sig==False].angle, ferret=ft, ax=ax[3], histtype='step')

        ax[0].set_xlabel(f'All Events: {len(df_)}')
        ax[1].set_xlabel(f'Significant Events: {len(df_[df_.sig==True])}')
        ax[2].set_xlabel(f'Non-Sig Events: {len(df_[df_.sig!=True])}')
        ax[3].set_xlabel('Comparison')

        ax[3].legend(['Sig', 'Non-Sig'])

        date = parse_directory_structure.get_all_dates_for_ferret(ft)[0]
        fdir = day_analysis_dir.format(ferret=ft, date=date)
        fdir = os.path.join(fdir, 'trajectories')
        fig.savefig(os.path.join(fdir, f'f{ft}_roseplot_sig_compare.png'))


def rose_and_comTrajectory():
    PFS_file_identifier=None
    seq_dir_identifier='nov23'
    save_flag=True


    ferrets = [261,317,335,336,337,339]
    tsers = [(2,15), None, None, (1,13), None, None]

    for nFerret in range(6):
        #get datasets
        ferret = ferrets[nFerret]
        tser = tsers[nFerret]


        date = parse_directory_structure.get_all_dates_for_ferret(ferret)[0]
        datasets = parse_directory_structure.get_all_datasets_for_date(ferret, date)
        if tser is None:
            datasets = [d for d in datasets if d.matchDesc('stim', descriptions.stim_type.none_black)]
        else:
            datasets = datasets[tser[0]:tser[1]]


        # % load in sequences

        #dict_events = sequence_metrics.main_streamlined(datasets, seq_dir_identifier=seq_dir_identifier, minFrames=4)
        dict_events = main_streamlined(datasets, seq_dir_identifier=seq_dir_identifier, minFrames=1)
        #df = dict_events['df']

        xcom, ycom = dict_events['xycom']
        xcom_diff = [np.diff(xcom_) for xcom_ in xcom]
        ycom_diff = [np.diff(ycom_) for ycom_ in ycom]
        xcom_diff=  np.concatenate(xcom_diff)
        ycom_diff = np.concatenate(ycom_diff)
        angle_all = [np.arctan2(-x,-y) for x,y in zip(xcom_diff,ycom_diff)]

        df_ = load_dataframe(ferrets[nFerret])['df']


        fig, ax = plt.subplots(1,2, subplot_kw={'projection': 'polar'}, figsize=(7,4))
        ax[0] = rose_plot(df_.angle, ferret, ax=ax[0])
        ax[1] = rose_plot(np.rad2deg(angle_all), ferret=ferret, ax=ax[1])

        ax[0].set_xlabel(f'Significantly Linear Sequences')
        ax[1].set_xlabel(f'Center of Mass Shifts--All Active Frames')

        date = parse_directory_structure.get_all_dates_for_ferret(ferret)[0]
        fdir = day_analysis_dir.format(ferret=ferret, date=date)
        fdir = os.path.join(fdir, 'trajectories')
        fig.savefig(os.path.join(fdir, f'F{ferret}_roseplot_com_compare.png'))
